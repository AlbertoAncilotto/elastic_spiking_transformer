import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Use clock_driven for SpikingJelly v0.0.0.0.12
from spikingjelly.clock_driven import neuron, functional
from timm.models import create_model

# Import your model definition
# Assuming 'model.py' is in the same directory or python path
import model
from ehw_gesture.ehwgesture import EHWGesture 

def parse_args():
    parser = argparse.ArgumentParser(description='Spike Analysis for Spikformer')
    # Model architecture args
    parser.add_argument('--patch-size', type=int, default=16, help='patch size')
    parser.add_argument('--embed-dims', type=int, default=256, help='embedding dimensions')
    parser.add_argument('--num-heads', type=int, default=32, help='number of attention heads')
    parser.add_argument('--mlp-ratios', type=int, default=4, help='MLP expansion ratio')
    parser.add_argument('--in-channels', type=int, default=2, help='input channels')
    parser.add_argument('--depths', type=int, default=2, help='number of transformer blocks')
    parser.add_argument('--sr-ratios', type=int, default=1, help='spatial reduction ratio')
    parser.add_argument('--sps-alpha', type=float, default=2.0, help='SPS alpha')
    parser.add_argument('--use-xisps', action='store_true', default=True)
    parser.add_argument('--xisps-elastic', action='store_true', default=True)
    parser.add_argument('--num-classes', type=int, default=22, help='number of classes')
    parser.add_argument('--attn-lower-heads-limit', type=int, default=2)
    parser.add_argument('--sps-lower-filter-limit', type=int, default=16)
    
    # Analysis args
    parser.add_argument('--checkpoint', default="logs/xisps2-elastic_alpha2.0_t32/spikformer_b8_T32_Ttrain32_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_max_test_acc1.pth", help='path to checkpoint')
    parser.add_argument('--model-name', default='default_model', help='model name for output directory')
    
    # Data args
    parser.add_argument('--data-path', default='data/ehwgesture/', help='dataset path')
    parser.add_argument('--T', default=16, type=int, help='simulation timesteps')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')

    return parser.parse_args()

def get_activation_hook(layer_name, storage_dict):
    """Helper to create a hook that stores the output spikes."""
    def hook(model, input, output):
        # Handle cases where output might be a tuple (spikes, v_mem)
        if isinstance(output, tuple):
            spikes = output[0]
        else:
            spikes = output
        
        # Detach and move to CPU to save memory
        storage_dict[layer_name] = spikes.detach().cpu()
    return hook

if __name__ == "__main__":
    args = parse_args()

    # Setup output directory
    output_dir = os.path.join('spike_analysis', args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file_path = os.path.join(output_dir, 'analysis_log.txt')
    
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file_path)
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file_path}")

    # 1. Simulation Parameters
    T = args.T   # Time steps
    N = args.batch_size    # Batch size
    granularities = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    
    # RECOMMENDED: Use CPU for analysis to avoid SpikingJelly 0.0.0.0.12 CUDA kernel errors
    device = 'cuda' 
    print(f"Running analysis on: {device.upper()}")

    # Load real data
    print("Loading EHWGesture dataset...")
    dataset = EHWGesture(
        root=args.data_path, 
        data_type='frame', 
        frames_number=T, 
        split_by='number'
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=N,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # 2. Create Model
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
        attn_lower_heads_limit=args.attn_lower_heads_limit,
        sps_lower_filter_limit=args.sps_lower_filter_limit
    )

    # 3. Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # Handle 'model' key or direct state_dict
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Load weights (strict=False often helps with minor version discrepancies)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. {msg}")
    else:
        print(f"[WARNING] Checkpoint file not found at {args.checkpoint}")
        print("Running with random weights...")

    model.to(device)
    model.eval()

    # 4. Manual Hook Registration (The Fix)
    # We use this instead of OutputMonitor because OutputMonitor failed to find your layers
    hooks = []
    neuron_count = 0

    print("--- Registering Hooks ---")
    for name, layer in model.named_modules():
        # Check for LIFNode in class name (handles MultiStepLIFNode, LIFNode, etc.)
        if "LIFNode" in layer.__class__.__name__:
            # print(f"Hooked: {name}") 
            neuron_count += 1

    if neuron_count == 0:
        print("[ERROR] No LIF Nodes found! Is the model definition correct?")
        sys.exit(1)
    else:
        print(f"Successfully identified {neuron_count} LIF layers.")

    # 5. Get real data batch (use first batch from dataset)
    # Input shape: [Time, Batch, Channels, Height, Width]
    print("\nLoading a batch of real data...")
    data_iter = iter(data_loader)
    x, labels = next(data_iter)
    x = x.float().to(device)
    print(f"Input shape: {x.shape}, Labels: {labels.tolist()}")

    # Get layer names in model definition order (not alphabetical)
    model_order_layer_names = [name for name, layer in model.named_modules() 
                                if "LIFNode" in layer.__class__.__name__]

    # Data structures to collect stats for plotting
    all_layer_names = model_order_layer_names  # Use model order
    spikes_per_granularity = {}  # {granularity_str: {layer_name: n_spikes}}
    rates_per_granularity = {}   # {granularity_str: {layer_name: firing_rate}}

    # 6. Run analysis for each granularity setting
    for granularity in granularities:
        print(f"\n{'='*100}")
        print(f"{'='*100}")
        print(f"GRANULARITY: {granularity}")
        print(f"{'='*100}")
        
        # Reset model state and counters
        functional.reset_net(model)
        layer_outputs = {}
        
        # Register hooks for this run
        hooks = []
        for name, layer in model.named_modules():
            if "LIFNode" in layer.__class__.__name__:
                h = layer.register_forward_hook(get_activation_hook(name, layer_outputs))
                hooks.append(h)
        
        # Run forward pass
        print(f"--- Running Forward Pass (T={T}) ---")
        with torch.no_grad():
            model(x, granularity)

        # Analyze Results
        print(f"\n{'Layer Name':<50} | {'Output Shape':<32} | {'Spikes':<8} | {'Firing Rate':<12}")
        print("-" * 100)

        total_spikes_model = 0
        total_elements_model = 0

        # Use model definition order (not alphabetical)
        ordered_layers = [name for name in model_order_layer_names if name in layer_outputs]

        for name in ordered_layers:
            spikes = layer_outputs[name]
            n_spikes = spikes.sum().item()
            if 'attn.k_lif' in name or 'attn.q_lif' in name or 'attn.v_lif' in name:
                # breakpoint()
                module = dict(model.named_modules())[name.replace('.k_lif', '') if 'k_lif' in name else name.replace('.q_lif', '') if 'q_lif' in name else name.replace('.v_lif', '')]
                num_heads = module.current_num_heads
                max_heads = module.max_num_heads
                n_spikes *= (num_heads/max_heads)
            if not ((spikes == 0.0) | (spikes == 1.0)).all():
                print(f"  WARNING: {name} contains non-binary values. Min: {spikes.min():.4f}, Max: {spikes.max():.4f}")
            n_elements = spikes.numel() # Total possible spikes
            rate = n_spikes / n_elements if n_elements > 0 else 0
            
            # Truncate long names for display
            display_name = (name[:47] + '..') if len(name) > 47 else name
            
            print(f"{display_name:<50} | {str(list(spikes.shape)):<32} | {int(n_spikes):<8} | {rate:.4f}")

            total_spikes_model += n_spikes
            total_elements_model += n_elements

        print("-" * 100)

        # Whole Model Stats
        model_rate = total_spikes_model / total_elements_model if total_elements_model > 0 else 0
        print(f"{'WHOLE MODEL per T per N':<50} | {int(total_elements_model/T/N):<32} | {int(total_spikes_model/T/N):<8} | {model_rate:.4f}")
        print(f"{'WHOLE MODEL PER N':<50} | {int(total_elements_model/N):<32} | {int(total_spikes_model/N):<8} | {model_rate:.4f}")

        # Store stats for plotting
        gran_key = str(granularity)
        spikes_per_granularity[gran_key] = {}
        rates_per_granularity[gran_key] = {}
        
        for name in ordered_layers:
            spikes = layer_outputs[name]
            n_spikes = spikes.sum().item()
            if 'attn.k_lif' in name or 'attn.q_lif' in name or 'attn.v_lif' in name:
                module = dict(model.named_modules())[name.replace('.k_lif', '') if 'k_lif' in name else name.replace('.q_lif', '') if 'q_lif' in name else name.replace('.v_lif', '')]
                num_heads = module.current_num_heads
                max_heads = module.max_num_heads
                n_spikes *= (num_heads/max_heads)
            n_elements = spikes.numel()
            rate = n_spikes / n_elements if n_elements > 0 else 0
            spikes_per_granularity[gran_key][name] = n_spikes
            rates_per_granularity[gran_key][name] = rate

        # Cleanup hooks for this run
        for h in hooks:
            h.remove()
        
        # Clear layer outputs to free memory
        layer_outputs.clear()

        params = model.get_granularity_parameters(granularity)
        print(f"\nGranularity Parameters: {params}")
    
    # Final cleanup
    functional.reset_net(model)

    # ===================== PLOTTING =====================
    print("\n" + "="*100)
    print("Generating Bar Plots...")
    print("="*100)

    # Prepare data for plotting
    n_layers = len(all_layer_names)
    n_granularities = len(granularities)
    x_positions = np.arange(n_layers)
    bar_width = 0.8 / n_granularities  # Divide available space among bars
    
    # Shorten layer names for display
    short_names = []
    for name in all_layer_names:
        # Extract meaningful part of the name
        parts = name.split('.')
        if len(parts) > 2:
            short_name = '.'.join(parts[-3:])  # Last 3 parts
        else:
            short_name = name
        if len(short_name) > 25:
            short_name = short_name[:22] + '..'
        short_names.append(short_name)

    colors = ["#1f7db4", "#4136d8", "#d01ad6", '#d62728']  # Blue, Orange, Green, Red
    granularity_labels = [str(g) for g in granularities]

    # ---- Plot 1: Spikes per Layer ----
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    
    for i, (gran_key, label) in enumerate(zip(spikes_per_granularity.keys(), granularity_labels)):
        spikes_values = [spikes_per_granularity[gran_key].get(name, 0) for name in all_layer_names]
        offset = (i - n_granularities/2 + 0.5) * bar_width
        ax1.bar(x_positions + offset, spikes_values, bar_width, label=f'Gran {label}', color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Number of Spikes', fontsize=12)
    ax1.set_title('Spikes per Layer at Different Granularities', fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.legend(title='Granularity', loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    spikes_plot_path = os.path.join(output_dir, 'spikes_per_layer.png')
    plt.savefig(spikes_plot_path, dpi=150)
    print(f"Saved: {spikes_plot_path}")
    plt.close(fig1)

    # ---- Plot 2: Firing Rate per Layer ----
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    
    for i, (gran_key, label) in enumerate(zip(rates_per_granularity.keys(), granularity_labels)):
        rate_values = [rates_per_granularity[gran_key].get(name, 0) for name in all_layer_names]
        offset = (i - n_granularities/2 + 0.5) * bar_width
        ax2.bar(x_positions + offset, rate_values, bar_width, label=f'Gran {label}', color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Firing Rate', fontsize=12)
    ax2.set_title('Firing Rate per Layer at Different Granularities', fontsize=14)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.legend(title='Granularity', loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    rates_plot_path = os.path.join(output_dir, 'firing_rate_per_layer.png')
    plt.savefig(rates_plot_path, dpi=150)
    print(f"Saved: {rates_plot_path}")
    plt.close(fig2)

    print("\nPlotting complete!")
    print(f"All outputs saved to: {output_dir}")