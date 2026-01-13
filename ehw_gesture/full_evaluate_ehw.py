import os
import time
import argparse

import torch
import torch.utils.data
from torch import nn
import model, utils
from spikingjelly.clock_driven import functional
from ehw_gesture.ehwgesture import EHWGesture   
from timm.models import create_model
import math
import numpy as np
import json


# Class mappings for the 22-class EHW Gesture dataset
CLASS_ID_TO_NAME = {
    0: 'FTF_L', 1: 'FTF_R', 2: 'FTN_L', 3: 'FTN_R', 4: 'FTS_L', 5: 'FTS_R',
    6: 'NOSE_L', 7: 'NOSE_R', 8: 'OCF_L', 9: 'OCF_R', 10: 'OCN_L', 11: 'OCN_R',
    12: 'OCS_L', 13: 'OCS_R', 14: 'PSF_L', 15: 'PSF_R', 16: 'PSN_L', 17: 'PSN_R',
    18: 'PSS_L', 19: 'PSS_R', 20: 'TR_L', 21: 'TR_R'
}

# Gesture type mapping (5 classes): FT=0, NOSE=1, OC=2, PS=3, TR=4
GESTURE_MAPPING = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,      # FT
    6: 1, 7: 1,                                # NOSE
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,   # OC
    14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, # PS
    20: 4, 21: 4                               # TR
}
GESTURE_NAMES = {0: 'FT', 1: 'NOSE', 2: 'OC', 3: 'PS', 4: 'TR'}

# Speed mapping (3 classes): F(fast)=0, N(normal)=1, S(slow)=2
# NOSE and TR are only normal speed (no suffix letter for speed)
SPEED_MAPPING = {
    0: 0, 1: 0,                               # FTF -> Fast
    2: 1, 3: 1,                               # FTN -> Normal
    4: 2, 5: 2,                               # FTS -> Slow
    6: 1, 7: 1,                               # NOSE -> Normal (default)
    8: 0, 9: 0,                               # OCF -> Fast
    10: 1, 11: 1,                             # OCN -> Normal
    12: 2, 13: 2,                             # OCS -> Slow
    14: 0, 15: 0,                             # PSF -> Fast
    16: 1, 17: 1,                             # PSN -> Normal
    18: 2, 19: 2,                             # PSS -> Slow
    20: 1, 21: 1                              # TR -> Normal (default)
}
SPEED_NAMES = {0: 'Fast', 1: 'Normal', 2: 'Slow'}

# Handedness mapping (2 classes): L=0, R=1
HANDEDNESS_MAPPING = {
    0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1,
    6: 0, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1,
    12: 0, 13: 1, 14: 0, 15: 1, 16: 0, 17: 1,
    18: 0, 19: 1, 20: 0, 21: 1
}
HANDEDNESS_NAMES = {0: 'Left', 1: 'Right'}


def map_to_grouped_classes(class_ids, mapping):
    """Map original class IDs to grouped class IDs using a mapping dict."""
    if isinstance(class_ids, torch.Tensor):
        device = class_ids.device
        mapped = torch.tensor([mapping[c.item()] for c in class_ids], device=device)
        return mapped
    return [mapping[c] for c in class_ids]


def compute_grouped_accuracy(output, target, mapping):
    """
    Compute accuracy after grouping classes.
    
    Takes the predicted class from the original 22-class output, maps it to 
    the grouped class, and compares with the mapped ground truth.
    
    Args:
        output: Model output logits [B, num_classes]
        target: Ground truth labels [B]
        mapping: Dict mapping original class -> grouped class
    
    Returns:
        Accuracy percentage
    """
    batch_size = output.shape[0]
    
    # Get the predicted class from original 22-class output
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)  # [B]
    
    # Map predictions to grouped classes
    grouped_pred = map_to_grouped_classes(pred, mapping)
    
    # Map targets to grouped classes
    grouped_target = map_to_grouped_classes(target, mapping)
    
    # Compute accuracy
    correct = grouped_pred.eq(grouped_target).sum().item()
    accuracy = 100.0 * correct / batch_size
    
    return accuracy


def split_to_train_test_set(train_ratio, origin_dataset, num_classes, random_split=False):
    """Split dataset into train and test sets with stratified sampling."""
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    
    train_idx = []
    test_idx = []
    
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def load_data(dataset_dir, T, num_classes, train_ratio=0.75, random_split=False):
    """Load EHW Gesture dataset and split into train/test sets."""
    print("Loading data...")
    st = time.time()

    origin_set = EHWGesture(
        root=dataset_dir, 
        data_type='frame', 
        frames_number=T, 
        split_by='number'
    )
    
    dataset_train, dataset_test = split_to_train_test_set(
        train_ratio, 
        origin_set, 
        num_classes,
        random_split=random_split
    )
    
    print(f"Loaded data in {time.time() - st:.2f}s")
    print(f"Test samples: {len(dataset_test)}")

    return dataset_test


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    """Evaluate model at granularity=0."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Accumulators for grouped accuracy
    gesture_correct, gesture_total = 0, 0
    speed_correct, speed_total = 0, 0
    hand_correct, hand_total = 0, 0
    
    with torch.no_grad():
        granularity = 3
        
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            
            output = model(image, granularity=granularity)
            loss = criterion(output, target)
            
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
            # Compute grouped accuracies
            gesture_acc = compute_grouped_accuracy(output, target, GESTURE_MAPPING)
            speed_acc = compute_grouped_accuracy(output, target, SPEED_MAPPING)
            hand_acc = compute_grouped_accuracy(output, target, HANDEDNESS_MAPPING)
            
            gesture_correct += gesture_acc * batch_size / 100.0
            gesture_total += batch_size
            speed_correct += speed_acc * batch_size / 100.0
            speed_total += batch_size
            hand_correct += hand_acc * batch_size / 100.0
            hand_total += batch_size

    metric_logger.synchronize_between_processes()
    
    loss = metric_logger.loss.global_avg
    acc1 = metric_logger.acc1.global_avg
    acc5 = metric_logger.acc5.global_avg
    
    # Compute final grouped accuracies
    gesture_acc_final = 100.0 * gesture_correct / gesture_total if gesture_total > 0 else 0.0
    speed_acc_final = 100.0 * speed_correct / speed_total if speed_total > 0 else 0.0
    hand_acc_final = 100.0 * hand_correct / hand_total if hand_total > 0 else 0.0
    
    print(f'\n{header} Results (Granularity {granularity}):')
    print(f'  Overall Acc@1: {acc1:.2f}%')
    print(f'  Overall Acc@5: {acc5:.2f}%')
    print(f'  Loss: {loss:.4f}')
    print(f'\n  Grouped Accuracies:')
    print(f'    Gesture (5 classes): {gesture_acc_final:.2f}%')
    print(f'    Speed (3 classes):   {speed_acc_final:.2f}%')
    print(f'    Handedness (2 classes): {hand_acc_final:.2f}%')
    
    return loss, acc1, acc5, gesture_acc_final, speed_acc_final, hand_acc_final


def full_evaluate(model, criterion, data_loader, device, args, print_freq=100):
    """Evaluate all 64 granularity combinations."""
    model.eval()
    print("\n" + "="*80)
    print("FULL GRANULARITY EVALUATION")
    print("Testing all 64 combinations: [Feat_Extractor, Attention, MLP]")
    print("="*80)
    
    results = {}
    
    with torch.no_grad():
        for g_feat in range(4):
            for g_attn in range(4):
                for g_mlp in range(4):
                    granularity = [g_feat, g_attn, g_mlp]
                    metric_logger = utils.MetricLogger(delimiter="  ")
                    
                    # Accumulators for grouped accuracy
                    gesture_correct, gesture_total = 0, 0
                    speed_correct, speed_total = 0, 0
                    hand_correct, hand_total = 0, 0
                    
                    header = f'Eval [F:{g_feat}, A:{g_attn}, M:{g_mlp}]'
                    for image, target in metric_logger.log_every(data_loader, print_freq, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        image = image.float()
                        
                        output = model(image, granularity=granularity)
                        loss = criterion(output, target)
                        
                        functional.reset_net(model)

                        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                        batch_size = image.shape[0]
                        metric_logger.update(loss=loss.item())
                        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                        
                        # Compute grouped accuracies
                        gesture_acc = compute_grouped_accuracy(output, target, GESTURE_MAPPING)
                        speed_acc = compute_grouped_accuracy(output, target, SPEED_MAPPING)
                        hand_acc = compute_grouped_accuracy(output, target, HANDEDNESS_MAPPING)
                        
                        gesture_correct += gesture_acc * batch_size / 100.0
                        gesture_total += batch_size
                        speed_correct += speed_acc * batch_size / 100.0
                        speed_total += batch_size
                        hand_correct += hand_acc * batch_size / 100.0
                        hand_total += batch_size

                    metric_logger.synchronize_between_processes()

                    parameters = model.get_granularity_parameters(granularity)
                    
                    loss = metric_logger.loss.global_avg
                    acc1 = metric_logger.acc1.global_avg
                    acc5 = metric_logger.acc5.global_avg
                    
                    # Compute final grouped accuracies
                    gesture_acc_final = 100.0 * gesture_correct / gesture_total if gesture_total > 0 else 0.0
                    speed_acc_final = 100.0 * speed_correct / speed_total if speed_total > 0 else 0.0
                    hand_acc_final = 100.0 * hand_correct / hand_total if hand_total > 0 else 0.0
                    
                    results[tuple(granularity)] = {
                        'loss': loss,
                        'acc1': acc1,
                        'acc5': acc5,
                        'parameters': parameters,
                        'gesture_acc': gesture_acc_final,
                        'speed_acc': speed_acc_final,
                        'handedness_acc': hand_acc_final,
                    }
                    
                    print(f'[F:{g_feat}, A:{g_attn}, M:{g_mlp}] - '
                          f'Acc@1: {acc1:.2f}%, Loss: {loss:.4f}, '
                          f'Params: {parameters:,}, '
                          f'Gesture: {gesture_acc_final:.1f}%, Speed: {speed_acc_final:.1f}%, Hand: {hand_acc_final:.1f}%')
    
    # Print summaries
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY ACCURACY")
    print("="*80)
    sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['acc1'], reverse=True)
    for i, (gran, metrics) in enumerate(sorted_by_acc[:10]):
        print(f"{i+1:2d}. [F:{gran[0]}, A:{gran[1]}, M:{gran[2]}]")
        print(f"    Acc@1: {metrics['acc1']:.2f}%, Loss: {metrics['loss']:.4f}")
        print(f"    Params: {metrics['parameters']:,}")
        print(f"    Gesture: {metrics['gesture_acc']:.2f}%, Speed: {metrics['speed_acc']:.2f}%, Handedness: {metrics['handedness_acc']:.2f}%")
    
    # Best configuration
    best_gran, best_metrics = sorted_by_acc[0]
    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: [F:{best_gran[0]}, A:{best_gran[1]}, M:{best_gran[2]}]")
    print(f"Acc@1: {best_metrics['acc1']:.2f}%, Acc@5: {best_metrics['acc5']:.2f}%")
    print(f"Loss: {best_metrics['loss']:.4f}, Params: {best_metrics['parameters']:,}")
    print(f"Gesture Acc: {best_metrics['gesture_acc']:.2f}%")
    print(f"Speed Acc: {best_metrics['speed_acc']:.2f}%")
    print(f"Handedness Acc: {best_metrics['handedness_acc']:.2f}%")
    print("="*80 + "\n")

    return results, best_gran, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Spikformer')
    
    # Model and dataset args
    parser.add_argument('--model', default='spikformer', help='model')
    parser.add_argument('--dataset', default='ehwgesture', help='dataset')
    parser.add_argument('--num-classes', type=int, default=22, help='number of classes')
    parser.add_argument('--data-path', default='data/ehwgesture/', help='dataset path')
    parser.add_argument('--device', default='cuda:9', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, help='data loading workers')
    parser.add_argument('--T', default=16, type=int, help='simulation timesteps')
    
    # Model architecture args
    parser.add_argument('--patch-size', type=int, default=16, help='patch size')
    parser.add_argument('--embed-dims', type=int, default=256, help='embedding dimensions')
    parser.add_argument('--num-heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--mlp-ratios', type=int, default=4, help='MLP expansion ratio')
    parser.add_argument('--in-channels', type=int, default=2, help='input channels')
    parser.add_argument('--depths', type=int, default=1, help='number of transformer blocks')
    parser.add_argument('--sr-ratios', type=int, default=1, help='spatial reduction ratio')
    parser.add_argument('--sps-alpha', type=float, default=1.0, help='SPS alpha')
    parser.add_argument('--use-xisps', action='store_true', default=False)
    parser.add_argument('--xisps-elastic', action='store_true', default=False)
    
    # Evaluation args
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint')
    parser.add_argument('--output-dir', default='./evaluation_results', help='output directory')
    parser.add_argument('--full-eval', action='store_true', default=False,
                        help='run full evaluation with all 64 granularity combinations')
    parser.add_argument('--train-ratio', type=float, default=0.75, help='train/test split ratio')
    parser.add_argument('--random-split', action='store_true', default=False)
    
    args = parser.parse_args()
    return args


def main(args):
    print("="*80)
    print("EVALUATION FOR SPIKFORMER")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timesteps: {args.T}")
    print("="*80 + "\n")
    
    # Setup device
    device = torch.device(args.device)
    
    # Create model
    print("Creating model...")
    model = create_model(
        'spikformer',
        pretrained=False,
        patch_size=args.patch_size,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratios,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        depths=args.depths,
        sr_ratios=args.sr_ratios,
        sps_alpha=args.sps_alpha,
        use_xisps=args.use_xisps,
        xisps_elastic=args.xisps_elastic,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f"Checkpoint loaded (epoch {checkpoint['epoch']})")
    print(f"Best test acc1 from training: {checkpoint.get('max_test_acc1', 'N/A')}")
    
    model.to(device)
    model.eval()
    
    # Load data
    dataset_test = load_data(
        args.data_path,
        args.T,
        args.num_classes,
        args.train_ratio,
        args.random_split
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80 + "\n")
    
    if args.full_eval:
        # Full evaluation with all 64 combinations
        results, best_gran, best_metrics = full_evaluate(
            model, criterion, data_loader_test, device, args
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, 'full_evaluation_results.json')
        with open(output_file, 'w') as f:
            json_results = {str(k): v for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        print(f"\nFull results saved to {output_file}")
        
    else:
        # Standard evaluation at granularity 0
        loss, acc1, acc5, gesture_acc, speed_acc, hand_acc = evaluate(
            model, criterion, data_loader_test, device
        )
        
        # Save results
        results = {
            'loss': loss,
            'acc1': acc1,
            'acc5': acc5,
            'gesture_acc': gesture_acc,
            'speed_acc': speed_acc,
            'handedness_acc': hand_acc,
        }
        
        output_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    main(args)