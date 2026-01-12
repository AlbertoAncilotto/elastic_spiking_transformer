import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchinfo
from torchvision import transforms
import math
from torch.cuda import amp
import model, utils
from spikingjelly.clock_driven import functional
from ehw_gesture.ehwgesture import EHWGesture  
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
import autoaugment
import numpy as np
import random
import wandb

root_path = os.path.abspath(__file__)
writer = SummaryWriter("./")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='spikformer', help='model')
    parser.add_argument('--dataset', default='ehwgesture', help='dataset')  # MODIFIED
    parser.add_argument('--num-classes', type=int, default=22, metavar='N',  
                        help='number of label classes (default: 22 for EHW Gesture)')
    parser.add_argument('--data-path', default='data/ehwgesture/', help='dataset')  # MODIFIED
    parser.add_argument('--device', default='cuda:9', help='device')  
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',  # MODIFIED: Reduced from 32
                        help='number of data loading workers (default: 8)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('--amp', default=False, action='store_true',
                        help='Use AMP training')

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True, action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')

    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=16, type=int)

    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.5)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    
    # WandB arguments
    parser.add_argument('--log-wandb', action='store_true', default=True,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb-project', type=str, default='spikformer-ehwgesture',  # MODIFIED
                        help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='wandb entity (team) name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='wandb run name (defaults to generated name)')
    
    # Spikformer model architecture arguments
    parser.add_argument('--patch-size', type=int, default=16,
                        help='patch size for spikformer (default: 16)')
    parser.add_argument('--embed-dims', type=int, default=256,
                        help='embedding dimensions (default: 256)')
    parser.add_argument('--num-heads', type=int, default=16,
                        help='number of attention heads (default: 16)')
    parser.add_argument('--mlp-ratios', type=int, default=4,
                        help='MLP expansion ratio (default: 4)')
    parser.add_argument('--in-channels', type=int, default=2,
                        help='number of input channels (default: 2 for DVS data)')
    parser.add_argument('--qkv-bias', action='store_true', default=False,
                        help='use bias in QKV projection')
    parser.add_argument('--depths', type=int, default=2,
                        help='number of transformer blocks (default: 2)')
    parser.add_argument('--sr-ratios', type=int, default=1,
                        help='spatial reduction ratio (default: 1)')
    parser.add_argument('--drop-rate', type=float, default=0.,
                        help='dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0.1,
                        help='drop path rate (default: 0.1)')
    parser.add_argument('--drop-block-rate', type=float, default=None,
                        help='drop block rate (default: None)')
    parser.add_argument('--sps-alpha', type=float, default=1.0,
                        help='SPS alpha (default: 1.0)')
    parser.add_argument('--use-xisps', action='store_true', default=False,
                        help='use smaller xisps patch splitting')
    parser.add_argument('--xisps-elastic', action='store_true', default=False,
                        help='make xisps patch splitting elastic too')
    parser.add_argument('--attn-lower-heads-limit', type=int, default=2,
                        help='minimum number of attention heads in granularities (default: 2)')
    parser.add_argument('--sps-lower-filter-limit', type=int, default=4,
                        help='minimum number of filters in SPS granularities (default: 4)')
    
    # NEW: Dataset split arguments for EHW Gesture
    parser.add_argument('--train-ratio', type=float, default=0.75,
                        help='ratio of training data (default: 0.75)')
    parser.add_argument('--random-split', action='store_true', default=False,
                        help='use random split instead of deterministic split')
    
    args = parser.parse_args()
    return args

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    """
    Split dataset into train and test sets with stratified sampling.
    
    Args:
        train_ratio: Ratio of training data (e.g., 0.9 for 90% train, 10% test)
        origin_dataset: Original dataset to split
        num_classes: Number of classes in the dataset
        random_split: Whether to randomly shuffle before splitting
    
    Returns:
        Tuple of (train_subset, test_subset)
    """
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    # Group samples by class
    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    
    train_idx = []
    test_idx = []
    
    # Optionally shuffle within each class
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    # Split each class
    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()
        N,T,C,H,W = image.shape
        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])
        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for granularity in range(4):
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
                metric_logger.meters[f'acc1_g{granularity}'].update(acc1.item(), n=batch_size)
                metric_logger.meters[f'acc5_g{granularity}'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    loss, acc1g0, acc1g3 = metric_logger.loss.global_avg, metric_logger.acc1_g0.global_avg, metric_logger.acc1_g3.global_avg
    print(f' * Acc@1 (G:0) = {acc1g0}, Acc@1 (G:3) = {acc1g3}, loss = {loss}')
    return loss, acc1g0, metric_logger.acc1_g1.global_avg, metric_logger.acc1_g2.global_avg, acc1g3


def full_evaluate(model, criterion, data_loader, device, print_freq=100):
    """
    Evaluate all combinations of granularity settings:
    granularity = [feat_extractor_gran, attn_gran, mlp_gran]
    Each can be 0, 1, 2, or 3 (4^3 = 64 combinations)
    """
    model.eval()
    print("\n" + "="*80)
    print("FULL GRANULARITY EVALUATION - Testing all 64 combinations")
    print("="*80)
    
    results = {}
    
    with torch.no_grad():
        for g_feat in range(4):
            for g_attn in range(4):
                for g_mlp in range(4):
                    granularity = [g_feat, g_attn, g_mlp]
                    metric_logger = utils.MetricLogger(delimiter="  ")
                    
                    header = f'Full Eval [F:{g_feat}, A:{g_attn}, M:{g_mlp}]'
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

                    metric_logger.synchronize_between_processes()

                    parameters = model.get_granularity_parameters(granularity)
                    
                    loss = metric_logger.loss.global_avg
                    acc1 = metric_logger.acc1.global_avg
                    acc5 = metric_logger.acc5.global_avg
                    
                    results[tuple(granularity)] = {
                        'loss': loss,
                        'acc1': acc1,
                        'acc5': acc5,
                        'parameters': parameters
                    }
                    
                    print(f'    Granularity Setting: {granularity}')
                    print(f'    Granularity Parameters: {parameters}')
                    print(f'[F:{g_feat}, A:{g_attn}, M:{g_mlp}] - Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%, Loss: {loss:.4f}')
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Top 10 Configurations by Acc@1")
    print("="*80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['acc1'], reverse=True)
    for i, (gran, metrics) in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. [F:{gran[0]}, A:{gran[1]}, M:{gran[2]}] - Acc@1: {metrics['acc1']:.2f}%, Acc@5: {metrics['acc5']:.2f}%, Loss: {metrics['loss']:.4f}")
        print(f"    Granularity Parameters: {metrics['parameters']}")
    
    print("\n" + "="*80)
    print("Bottom 10 Configurations by Acc@1")
    print("="*80)
    for i, (gran, metrics) in enumerate(sorted_results[-10:]):
        print(f"{i+1:2d}. [F:{gran[0]}, A:{gran[1]}, M:{gran[2]}] - Acc@1: {metrics['acc1']:.2f}%, Acc@5: {metrics['acc5']:.2f}%, Loss: {metrics['loss']:.4f}")
        print(f"    Granularity Parameters: {metrics['parameters']}")
    
    # Find best configuration
    best_gran, best_metrics = sorted_results[0]
    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: [F:{best_gran[0]}, A:{best_gran[1]}, M:{best_gran[2]}]")
    print(f"Acc@1: {best_metrics['acc1']:.2f}%, Acc@5: {best_metrics['acc5']:.2f}%, Loss: {best_metrics['loss']:.4f}")
    print(f"    Granularity Parameters: {best_metrics['parameters']}")
    print("="*80 + "\n")
    
    return results, best_gran, best_metrics


def load_data(dataset_dir, distributed, T, num_classes, train_ratio=0.9, random_split=False):
    """
    Load EHW Gesture dataset and split into train/test sets.
    
    Args:
        dataset_dir: Root directory of the dataset
        distributed: Whether using distributed training
        T: Number of temporal frames
        num_classes: Number of classes in dataset
        train_ratio: Ratio of training data
        random_split: Whether to use random split
    
    Returns:
        Tuple of (dataset_train, dataset_test, train_sampler, test_sampler)
    """
    print("Loading data")
    st = time.time()

    # MODIFIED: Use EHWGesture instead of CIFAR10DVS
    origin_set = EHWGesture(
        root=dataset_dir, 
        data_type='frame', 
        frames_number=T, 
        split_by='number'
    )
    
    # Split into train and test
    dataset_train, dataset_test = split_to_train_test_set(
        train_ratio, 
        origin_set, 
        num_classes,
        random_split=random_split
    )
    
    print(f"Took {time.time() - st:.2f}s")
    print(f"Training samples: {len(dataset_train)}")
    print(f"Testing samples: {len(dataset_test)}")

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)
    data_path = args.data_path
    
    # MODIFIED: Pass num_classes and split parameters
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        data_path, 
        args.distributed, 
        args.T,
        args.num_classes,
        args.train_ratio,
        args.random_split
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True)

    model = create_model(
        'spikformer',
        pretrained=False,
        patch_size=args.patch_size,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratios,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        qkv_bias=args.qkv_bias,
        depths=args.depths,
        sr_ratios=args.sr_ratios,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        drop_block_rate=args.drop_block_rate,
        sps_alpha=args.sps_alpha,
        use_xisps=args.use_xisps,
        xisps_elastic=args.xisps_elastic,
        attn_lower_heads_limit=args.attn_lower_heads_limit,
        sps_lower_filter_limit=args.sps_lower_filter_limit,
    )

    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    print("\n\n ================== Model Summary: ================== \n\n")
    torchinfo.summary(model)

    print("\n\n ================== Patch Embedding Summary: ================== \n\n")
    torchinfo.summary(model.patch_embed)

    print("\n\n ================== Single Block Summary: ================== \n\n")
    torchinfo.summary(model.block)

    print("\n\n ================== CLS Summary: ================== \n\n")
    torchinfo.summary(model.head)

    model.get_granularity_info()
    
    # Initialize WandB
    if args.log_wandb and utils.is_main_process():
        # Create run name if not provided
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"{args.model}_b{args.batch_size}_T{args.T}_lr{args.lr}"
            if args.T_train:
                run_name += f"_Ttrain{args.T_train}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            resume='allow',
            id=run_name if args.resume else None
        )
        
        # Log model info to wandb
        wandb.config.update({"n_parameters": n_parameters})
    
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        full_evaluate(model, criterion, data_loader_test, device=device)
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    
    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= 150:
            mixup_fn.mixup_enabled = False
        
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        
        if utils.is_main_process():
            # TensorBoard logging
            if train_tb_writer is not None:
                train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            
            # WandB logging
            if args.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/acc1': train_acc1,
                    'train/acc5': train_acc5,
                    'train/lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)
        
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1g0, test_acc1g1, test_acc1g2, test_acc1g3 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        
        if utils.is_main_process():
            # TensorBoard logging
            if te_tb_writer is not None:
                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1_g0', test_acc1g0, epoch)
                te_tb_writer.add_scalar('test_acc1_g1', test_acc1g1, epoch)
                te_tb_writer.add_scalar('test_acc1_g2', test_acc1g2, epoch)
                te_tb_writer.add_scalar('test_acc1_g3', test_acc1g3, epoch)
            
            # WandB logging
            if args.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'test/loss': test_loss,
                    'test/acc1': test_acc1g0,
                    'test/acc1_g1': test_acc1g1,
                    'test/acc1_g2': test_acc1g2,
                    'test/acc1_g3': test_acc1g3,
                }, step=epoch)

        if max_test_acc1 < test_acc1g3:
            max_test_acc1 = test_acc1g3
            test_acc5_at_max_test_acc1 = test_acc1g0
            save_max = True
            
            # Log best metrics to wandb
            if args.log_wandb and utils.is_main_process():
                wandb.run.summary["best_test_acc1"] = max_test_acc1
                wandb.run.summary["best_test_acc5"] = test_acc5_at_max_test_acc1
                wandb.run.summary["best_epoch"] = epoch

        if output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                
                # Save best model to wandb
                if args.log_wandb and utils.is_main_process():
                    wandb.save(os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
    
    # Run full evaluation with all granularity combinations
    print("\n\nRunning full granularity evaluation...")
    full_results, best_gran, best_metrics = full_evaluate(
        model, criterion, data_loader_test, device=device)
    
    # Log full evaluation results to wandb
    if args.log_wandb and utils.is_main_process():
        # Log best configuration
        wandb.run.summary["best_full_eval_granularity"] = f"F:{best_gran[0]}, A:{best_gran[1]}, M:{best_gran[2]}"
        wandb.run.summary["best_full_eval_acc1"] = best_metrics['acc1']
        wandb.run.summary["best_full_eval_acc5"] = best_metrics['acc5']
        wandb.run.summary["best_full_eval_loss"] = best_metrics['loss']
        
        # Create a table with all results
        full_eval_table = wandb.Table(columns=["Feat_Gran", "Attn_Gran", "MLP_Gran", "Acc@1", "Acc@5", "Loss"])
        for gran, metrics in full_results.items():
            full_eval_table.add_data(gran[0], gran[1], gran[2], 
                                    metrics['acc1'], metrics['acc5'], metrics['loss'])
        wandb.log({"full_evaluation_results": full_eval_table})
    
    # Finish wandb run
    if args.log_wandb and utils.is_main_process():
        wandb.finish()

    return max_test_acc1

if __name__ == "__main__":
    args = parse_args()
    main(args)