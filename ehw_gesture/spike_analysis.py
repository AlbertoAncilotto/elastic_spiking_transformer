import datetime
import os
import time
import argparse

import torch
import torch.utils.data
from torch import nn
import torchinfo
from torch.cuda import amp
import model, utils
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from ehw_gesture.ehwgesture import EHWGesture   
from timm.models import create_model
import math
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt  # Added for plotting

class SpikeCounter:
    """Counts spikes from MultiStepLIFNode neurons during forward pass."""
    
    def __init__(self):
        self.spike_counts_per_timestep = []  # List of tensors [T] for each layer (per sample)
        self.spike_counts_per_layer = []     # Total spikes per layer (per sample)
        self.output_units_per_layer = []     # Number of output units per layer
        self.layer_names = []
        self.hooks = []
        self.T = None
    
    def register_hooks(self, model):
        """Register hooks on all MultiStepLIFNode modules."""
        for name, module in model.named_modules():
            if isinstance(module, MultiStepLIFNode):
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
                self.layer_names.append(name)
    
    def _create_hook(self, layer_name):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if output is not None and torch.is_tensor(output):
                # Output shape is typically [T, B, ...] for MultiStepLIFNode
                # breakpoint()
                B = output.shape[1] if output.dim() > 1 else 1
                
                # Calculate number of output units per sample (all dims except T and B)
                # For example, [T, B, C, N] -> C * N units per sample
                breakpoint()
                units_per_sample = output[0, 0].numel() if B > 0 else output[0].numel()
                
                # Sum over all dimensions except T and B to get spikes per sample
                # Shape after sum: [T, B]
                dims_to_sum = list(range(2, output.dim()))
                if dims_to_sum:
                    spike_count_per_t_per_batch = output.sum(dim=dims_to_sum)  # [T, B]
                else:
                    spike_count_per_t_per_batch = output  # Already [T, B] or similar
                
                # Average across batch dimension to get per-sample spikes per timestep
                spike_count_per_t = spike_count_per_t_per_batch.mean(dim=1)  # [T]
                
                # Store timestep-wise counts (per sample)
                self.spike_counts_per_timestep.append(spike_count_per_t.detach().cpu())
                
                # Store total count for this layer (per sample)
                total_spikes = spike_count_per_t.sum().item()
                self.spike_counts_per_layer.append(total_spikes)
                
                breakpoint()
                # Store output units
                self.output_units_per_layer.append(units_per_sample)
                
                if self.T is None:
                    self.T = output.shape[0]
        
        return hook
    
    def get_counts_per_timestep(self):
        """
        Aggregate spike counts across all layers for each timestep (per sample).
        Returns: Tensor of shape [T] with total spikes per timestep per sample
        """
        if not self.spike_counts_per_timestep:
            return None
        
        # Stack all layer counts: [num_layers, T]
        stacked = torch.stack(self.spike_counts_per_timestep, dim=0)
        # Sum across layers: [T]
        return stacked.sum(dim=0)
    
    def get_layer_statistics(self):
        """
        Get detailed statistics per layer including sparsity.
        Returns: Dictionary with layer-wise spike information
        """
        stats = {}
        for i, (name, total, units) in enumerate(zip(
            self.layer_names, 
            self.spike_counts_per_layer,
            self.output_units_per_layer
        )):
            if i < len(self.spike_counts_per_timestep):
                per_t = self.spike_counts_per_timestep[i]
                
                # Calculate sparsity: 1 - (spikes / total_possible_spikes)
                # Total possible spikes = units * T
                total_possible_spikes = units * self.T
                sparsity = 1.0 - (total / total_possible_spikes) if total_possible_spikes > 0 else 1.0
                
                # Firing rate: spikes / total_possible_spikes
                firing_rate = total / total_possible_spikes if total_possible_spikes > 0 else 0.0
                
                stats[name] = {
                    'total_spikes': total,
                    'output_units': units,
                    'total_possible_spikes': total_possible_spikes,
                    'sparsity': sparsity,
                    'firing_rate': firing_rate,
                    'mean_per_timestep': per_t.mean().item(),
                    'std_per_timestep': per_t.std().item(),
                    'min_per_timestep': per_t.min().item(),
                    'max_per_timestep': per_t.max().item(),
                }
        return stats
    
    def reset(self):
        """Reset all counters."""
        self.spike_counts_per_timestep = []
        self.spike_counts_per_layer = []
        self.output_units_per_layer = []
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


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


def evaluate_with_spikes(model, criterion, data_loader, device, spike_counter, print_freq=100, header='Test:'):
    """Evaluate model with spike counting for granularity=0."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Storage for spike statistics (averaged per sample)
    all_timestep_spikes = []
    all_layer_stats = []
    
    with torch.no_grad():
        granularity = 0  # Only evaluate at granularity 0
        
        for batch_idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            
            # Reset spike counter for this batch
            spike_counter.reset()
            
            # Forward pass
            output = model(image, granularity=granularity)
            loss = criterion(output, target)
            
            # Get spike statistics (already per-sample from counter)
            timestep_spikes = spike_counter.get_counts_per_timestep()
            layer_stats = spike_counter.get_layer_statistics()
            
            if timestep_spikes is not None:
                all_timestep_spikes.append(timestep_spikes)
                all_layer_stats.append(layer_stats)
            
            # Reset network state
            functional.reset_net(model)

            # Calculate accuracy
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    # Aggregate spike statistics across all batches (average per sample)
    avg_timestep_spikes = torch.stack(all_timestep_spikes).mean(dim=0)
    
    # Aggregate layer statistics (average per sample)
    aggregated_layer_stats = defaultdict(lambda: defaultdict(list))
    for batch_stats in all_layer_stats:
        for layer_name, stats in batch_stats.items():
            for key, value in stats.items():
                aggregated_layer_stats[layer_name][key].append(value)
    
    # Average the layer statistics
    final_layer_stats = {}
    for layer_name, stats_dict in aggregated_layer_stats.items():
        final_layer_stats[layer_name] = {
            key: np.mean(values) for key, values in stats_dict.items()
        }
    
    loss = metric_logger.loss.global_avg
    acc1 = metric_logger.acc1.global_avg
    acc5 = metric_logger.acc5.global_avg
    
    # Calculate total statistics and Split Statistics
    total_spikes = avg_timestep_spikes.sum().item()
    feat_extractor_spikes = 0
    transformer_spikes = 0
    
    for layer_name, stats in final_layer_stats.items():
        s_count = stats['total_spikes']
        if "patch_embed" in layer_name:
            feat_extractor_spikes += s_count
        elif "block" in layer_name and ("attn" in layer_name or "mlp" in layer_name):
            transformer_spikes += s_count

    total_units = sum(s['output_units'] for s in final_layer_stats.values())
    total_possible_spikes = total_units * spike_counter.T
    overall_sparsity = 1.0 - (total_spikes / total_possible_spikes) if total_possible_spikes > 0 else 1.0
    overall_firing_rate = total_spikes / total_possible_spikes if total_possible_spikes > 0 else 0.0
    
    print(f'\n{header} Results (Granularity 0):')
    print(f'  Acc@1: {acc1:.2f}%')
    print(f'  Acc@5: {acc5:.2f}%')
    print(f'  Loss: {loss:.4f}')
    print(f'\nOverall Spike Statistics (per sample):')
    print(f'  Total spikes: {total_spikes:.2f}')
    print(f'  - Feature Extractor Spikes: {feat_extractor_spikes:.2f}')
    print(f'  - Transformer Spikes: {transformer_spikes:.2f}')
    print(f'  Total output units: {total_units:,}')
    print(f'  Total possible spikes: {total_possible_spikes:,}')
    print(f'  Overall sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)')
    print(f'  Overall firing rate: {overall_firing_rate:.4f} ({overall_firing_rate*100:.2f}%)')
    print(f'  Mean spikes per timestep: {avg_timestep_spikes.mean().item():.2f}')
    print(f'  Std spikes per timestep: {avg_timestep_spikes.std().item():.2f}')
    
    return loss, acc1, acc5, avg_timestep_spikes, final_layer_stats, feat_extractor_spikes, transformer_spikes


def full_evaluate_with_spikes(model, criterion, data_loader, device, spike_counter, args, print_freq=100):
    """Evaluate all 64 granularity combinations with spike counting."""
    model.eval()
    print("\n" + "="*80)
    print("FULL GRANULARITY EVALUATION WITH SPIKE ANALYSIS")
    print("Testing all 64 combinations: [Feat_Extractor, Attention, MLP]")
    print("="*80)
    
    results = {}
    
    with torch.no_grad():
        for g_feat in range(4):
            for g_attn in range(4):
                for g_mlp in range(4):
                    granularity = [g_feat, g_attn, g_mlp]
                    metric_logger = utils.MetricLogger(delimiter="  ")
                    
                    # Storage for spike statistics (per sample)
                    all_timestep_spikes = []
                    all_layer_stats = []
                    
                    header = f'Eval [F:{g_feat}, A:{g_attn}, M:{g_mlp}]'
                    for image, target in metric_logger.log_every(data_loader, print_freq, header):
                        image = image.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        image = image.float()
                        
                        # Reset spike counter
                        spike_counter.reset()
                        
                        # Forward pass
                        output = model(image, granularity=granularity)
                        loss = criterion(output, target)
                        
                        # Get spike statistics (per sample)
                        timestep_spikes = spike_counter.get_counts_per_timestep()
                        layer_stats = spike_counter.get_layer_statistics()
                        
                        if timestep_spikes is not None:
                            all_timestep_spikes.append(timestep_spikes)
                            all_layer_stats.append(layer_stats)
                        
                        # Reset network
                        functional.reset_net(model)

                        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                        batch_size = image.shape[0]
                        metric_logger.update(loss=loss.item())
                        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                    metric_logger.synchronize_between_processes()

                    # Aggregate spike statistics (per sample)
                    avg_timestep_spikes = torch.stack(all_timestep_spikes).mean(dim=0)
                    total_spikes = avg_timestep_spikes.sum().item()
                    mean_spikes_per_t = avg_timestep_spikes.mean().item()
                    
                    # Aggregate layer statistics
                    aggregated_layer_stats = defaultdict(lambda: defaultdict(list))
                    for batch_stats in all_layer_stats:
                        for layer_name, stats in batch_stats.items():
                            for key, value in stats.items():
                                aggregated_layer_stats[layer_name][key].append(value)
                    
                    # Average the layer statistics
                    final_layer_stats = {}
                    for layer_name, stats_dict in aggregated_layer_stats.items():
                        final_layer_stats[layer_name] = {
                            key: np.mean(values) for key, values in stats_dict.items()
                        }
                    
                    # --- NEW SEPARATION LOGIC ---
                    feat_extractor_spikes = 0
                    transformer_spikes = 0
                    
                    for layer_name, stats in final_layer_stats.items():
                        s_count = stats['total_spikes']
                        
                        # Feature extractor condition
                        if "patch_embed" in layer_name:
                            feat_extractor_spikes += s_count
                        
                        # Transformer condition
                        elif "block" in layer_name and ("attn" in layer_name or "mlp" in layer_name):
                            transformer_spikes += s_count
                    # ----------------------------

                    # Calculate overall sparsity
                    # breakpoint()
                    total_units = sum(s['output_units'] for s in final_layer_stats.values())
                    total_possible_spikes = total_units * spike_counter.T
                    overall_sparsity = 1.0 - (total_spikes / total_possible_spikes) if total_possible_spikes > 0 else 1.0
                    
                    # Get parameter count for this granularity
                    parameters = model.get_granularity_parameters(granularity)
                    
                    loss = metric_logger.loss.global_avg
                    acc1 = metric_logger.acc1.global_avg
                    acc5 = metric_logger.acc5.global_avg
                    
                    results[tuple(granularity)] = {
                        'loss': loss,
                        'acc1': acc1,
                        'acc5': acc5,
                        'parameters': parameters,
                        'total_spikes': total_spikes,
                        'feat_extractor_spikes': feat_extractor_spikes, # Stored
                        'transformer_spikes': transformer_spikes,       # Stored
                        'total_units': total_units,
                        'overall_sparsity': overall_sparsity,
                        'mean_spikes_per_timestep': mean_spikes_per_t,
                        'timestep_spikes': avg_timestep_spikes.numpy().tolist(),
                        'layer_stats': {k: dict(v) for k, v in final_layer_stats.items()},
                    }
                    
                    print(f'[F:{g_feat}, A:{g_attn}, M:{g_mlp}] - '
                          f'Acc@1: {acc1:.2f}%, Loss: {loss:.4f}, '
                          f'Params: {parameters:,}, '
                          f'Total Spikes: {total_spikes:.0f} (Feat: {feat_extractor_spikes:.0f}, Trans: {transformer_spikes:.0f}), '
                          f'Sparsity: {overall_sparsity:.4f}')
    
    # Print summaries
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY ACCURACY")
    print("="*80)
    sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['acc1'], reverse=True)
    for i, (gran, metrics) in enumerate(sorted_by_acc[:10]):
        print(f"{i+1:2d}. [F:{gran[0]}, A:{gran[1]}, M:{gran[2]}]")
        print(f"    Acc@1: {metrics['acc1']:.2f}%, Loss: {metrics['loss']:.4f}")
        print(f"    Params: {metrics['parameters']:,}, Total Spikes: {metrics['total_spikes']:.0f}")
        print(f"    (Feat: {metrics['feat_extractor_spikes']:.0f}, Trans: {metrics['transformer_spikes']:.0f})")
        print(f"    Sparsity: {metrics['overall_sparsity']:.4f}")
    
    # Best configuration
    best_gran, best_metrics = sorted_by_acc[0]
    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: [F:{best_gran[0]}, A:{best_gran[1]}, M:{best_gran[2]}]")
    print(f"Acc@1: {best_metrics['acc1']:.2f}%, Acc@5: {best_metrics['acc5']:.2f}%")
    print(f"Loss: {best_metrics['loss']:.4f}, Params: {best_metrics['parameters']:,}")
    print(f"Total Spikes: {best_metrics['total_spikes']:.0f}")
    print(f"Feature Extractor Spikes: {best_metrics['feat_extractor_spikes']:.0f}")
    print(f"Transformer Spikes: {best_metrics['transformer_spikes']:.0f}")
    print(f"Overall Sparsity: {best_metrics['overall_sparsity']:.4f} ({best_metrics['overall_sparsity']*100:.2f}%)")
    print("="*80 + "\n")
    
    # --- PLOTTING LOGIC ---
    print("Generating Scatter Plots...")
    
    accuracies = [m['acc1'] for m in results.values()]
    total_spikes_list = [m['total_spikes'] for m in results.values()]
    transformer_spikes_list = [m['transformer_spikes'] for m in results.values()]

    # Plot 1: Accuracy vs Total Spikes
    plt.figure(figsize=(10, 6))
    plt.scatter(total_spikes_list, accuracies, alpha=0.7, c='b', edgecolors='k')
    plt.title('Accuracy vs. Total Spikes')
    plt.xlabel('Total Spikes (per sample)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_total = os.path.join(args.output_dir, 'scatter_acc_vs_total_spikes.png')
    plt.savefig(plot_path_total)
    plt.close()
    print(f"Saved plot: {plot_path_total}")

    # Plot 2: Accuracy vs Transformer Spikes
    plt.figure(figsize=(10, 6))
    plt.scatter(transformer_spikes_list, accuracies, alpha=0.7, c='r', edgecolors='k')
    plt.title('Accuracy vs. Transformer Spikes')
    plt.xlabel('Transformer Spikes (per sample)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_trans = os.path.join(args.output_dir, 'scatter_acc_vs_transformer_spikes.png')
    plt.savefig(plot_path_trans)
    plt.close()
    print(f"Saved plot: {plot_path_trans}")
    # ----------------------

    return results, best_gran, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Spike Analysis for Spikformer')
    
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
    
    # Analysis args
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint')
    parser.add_argument('--output-dir', default='./spike_analysis_results', help='output directory')
    parser.add_argument('--full-eval', action='store_true', default=False,
                        help='run full evaluation with all 64 granularity combinations')
    parser.add_argument('--train-ratio', type=float, default=0.75, help='train/test split ratio')
    parser.add_argument('--random-split', action='store_true', default=False)
    
    args = parser.parse_args()
    return args

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main(args):
    print("="*80)
    print("SPIKE ANALYSIS FOR SPIKFORMER")
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

    # Create spike counter and register hooks
    spike_counter = SpikeCounter()
    spike_counter.register_hooks(model)
    print(f"\nRegistered hooks on {len(spike_counter.layer_names)} LIF layers:")
    for name in spike_counter.layer_names:
        print(f"  - {name}")
    
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
        # Modified to accept args for plot saving path
        results, best_gran, best_metrics = full_evaluate_with_spikes(
            model, criterion, data_loader_test, device, spike_counter, args
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, 'full_evaluation_results.json')
        with open(output_file, 'w') as f:
            # Convert tuple keys to strings for JSON
            json_results = {str(k): to_python(v) for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        print(f"\nFull results saved to {output_file}")
        
    else:
        # Standard evaluation at granularity 0
        loss, acc1, acc5, timestep_spikes, layer_stats, feat_spikes, trans_spikes = evaluate_with_spikes(
            model, criterion, data_loader_test, device, spike_counter
        )
        
        # Calculate totals for summary
        total_spikes = timestep_spikes.sum().item()
        total_units = sum(s['output_units'] for s in layer_stats.values())
        total_possible_spikes = total_units * spike_counter.T
        overall_sparsity = 1.0 - (total_spikes / total_possible_spikes)
        
        # Save results
        results = {
            'loss': loss,
            'acc1': acc1,
            'acc5': acc5,
            'timestep_spikes': timestep_spikes.numpy().tolist(),
            'layer_statistics': layer_stats,
            'total_spikes': total_spikes,
            'feat_extractor_spikes': feat_spikes,
            'transformer_spikes': trans_spikes,
            'total_units': total_units,
            'total_possible_spikes': total_possible_spikes,
            'overall_sparsity': overall_sparsity,
        }
        
        output_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(to_python(results), f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Print layer statistics with sparsity
        print("\n" + "="*80)
        print("LAYER-WISE SPIKE STATISTICS (per sample)")
        print("="*80)
        for layer_name, stats in layer_stats.items():
            print(f"\n{layer_name}:")
            print(f"  Total spikes: {stats['total_spikes']:.2f}")
            print(f"  Output units: {stats['output_units']:,}")
            print(f"  Total possible spikes: {stats['total_possible_spikes']:,}")
            print(f"  Sparsity: {stats['sparsity']:.4f} ({stats['sparsity']*100:.2f}%)")
            print(f"  Firing rate: {stats['firing_rate']:.4f} ({stats['firing_rate']*100:.2f}%)")
            print(f"  Mean per timestep: {stats['mean_per_timestep']:.2f}")
            print(f"  Std per timestep: {stats['std_per_timestep']:.2f}")
    
    # Clean up
    spike_counter.remove_hooks()
    print("\n" + "="*80)
    print("SPIKE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    main(args)