"""
Scaling Analysis: Firing Rate vs Granularity
Analyzes linear, sublinear, superlinear, and inverse scaling patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create output directory
output_dir = 'xispikeformer_t16'
os.makedirs(output_dir, exist_ok=True)

# Data extracted from analysis_log.txt for XiSpikformer
# Format: {layer_name: [g0, g1, g2, g3]} - firing rates
firing_rates = {
    # Patch Embedding layers
    'patch_embed.proj_lif': [0.0034, 0.0051, 0.0086, 0.0137],
    'patch_embed.proj_lif1': [0.0076, 0.0153, 0.0305, 0.0611],
    'patch_embed.proj_lif2': [0.0018, 0.0036, 0.0071, 0.0142],
    'patch_embed.proj_lif3': [0.0015, 0.0045, 0.0103, 0.0212],
    'patch_embed.rpe_lif': [0.0089, 0.0176, 0.0392, 0.0819],
    # Block 0 Attention
    'block.0.attn.q_lif': [0.0175, 0.0458, 0.1014, 0.1724],
    'block.0.attn.k_lif': [0.0115, 0.0215, 0.0406, 0.0575],
    'block.0.attn.v_lif': [0.0083, 0.0192, 0.0363, 0.0521],
    'block.0.attn.attn_lif': [0.0166, 0.0588, 0.0901, 0.1028],
    'block.0.attn.proj_lif': [0.0938, 0.1056, 0.1170, 0.1206],
    # Block 0 MLP
    'block.0.mlp.fc1_lif': [0.0769, 0.0756, 0.0605, 0.0296],
    'block.0.mlp.fc2_lif': [0.1333, 0.0930, 0.0738, 0.0746],
    # Block 1 Attention
    'block.1.attn.q_lif': [0.0177, 0.0438, 0.0689, 0.1195],
    'block.1.attn.k_lif': [0.0120, 0.0231, 0.0426, 0.0577],
    'block.1.attn.v_lif': [0.0111, 0.0219, 0.0389, 0.0536],
    'block.1.attn.attn_lif': [0.0100, 0.0319, 0.0421, 0.0483],
    'block.1.attn.proj_lif': [0.0778, 0.1815, 0.1656, 0.1559],
    # Block 1 MLP
    'block.1.mlp.fc1_lif': [0.2096, 0.1366, 0.0800, 0.0414],
    'block.1.mlp.fc2_lif': [0.3552, 0.2571, 0.1278, 0.1131],
}

# Spike counts for energy analysis
spike_counts = {
    'patch_embed.proj_lif': [1437588, 2156382, 3593971, 5750354],
    'patch_embed.proj_lif1': [1601283, 3202566, 6405133, 12810266],
    'patch_embed.proj_lif2': [186747, 373495, 746990, 1493980],
    'patch_embed.proj_lif3': [38638, 117210, 269158, 556992],
    'patch_embed.rpe_lif': [58604, 115017, 257163, 536756],
    'block.0.attn.q_lif': [114693, 299907, 664706, 1130040],
    'block.0.attn.k_lif': [75156, 140728, 265848, 376894],
    'block.0.attn.v_lif': [54547, 126132, 237738, 341193],
    'block.0.attn.attn_lif': [108706, 385273, 590529, 673539],
    'block.0.attn.proj_lif': [614502, 691901, 766613, 790055],
    'block.0.mlp.fc1_lif': [126072, 309515, 643820, 776316],
    'block.0.mlp.fc2_lif': [873856, 609793, 483601, 488880],
    'block.1.attn.q_lif': [115816, 287061, 451612, 783461],
    'block.1.attn.k_lif': [78851, 151667, 278945, 378070],
    'block.1.attn.v_lif': [73000, 143844, 255010, 351591],
    'block.1.attn.attn_lif': [65334, 209017, 275939, 316366],
    'block.1.attn.proj_lif': [509953, 1189675, 1084951, 1021863],
    'block.1.mlp.fc1_lif': [343401, 559650, 851667, 1085237],
    'block.1.mlp.fc2_lif': [2327743, 1684963, 837395, 741317],
}

# MLP hidden widths at each granularity
mlp_widths = [64, 160, 416, 1024]
# Attention heads at each granularity  
attn_heads = [8, 16, 24, 32]
# Granularity labels
granularities = [0, 1, 2, 3]

# Parameters at each granularity
parameters = [680502, 918694, 1460998, 2587094]

def compute_scaling_exponent(values):
    """Compute scaling exponent using log-log regression"""
    x = np.array([0, 1, 2, 3])
    y = np.array(values)
    
    # Handle zeros and negative values
    if np.any(y <= 0):
        return None, None
    
    log_x = x + 1  # Shift to avoid log(0)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    return slope, r_value**2

def classify_scaling(slope):
    """Classify scaling type based on exponent"""
    if slope is None:
        return "undefined"
    elif slope < -0.1:
        return "inverse"
    elif slope < 0.5:
        return "sublinear"
    elif slope < 1.5:
        return "linear"
    else:
        return "superlinear"

# Analyze scaling for each layer
print("="*80)
print("SCALING ANALYSIS: Firing Rate vs Granularity")
print("="*80)

scaling_results = {}
for layer, rates in firing_rates.items():
    slope, r2 = compute_scaling_exponent(rates)
    scaling_type = classify_scaling(slope)
    scaling_results[layer] = {
        'rates': rates,
        'slope': slope,
        'r2': r2,
        'type': scaling_type,
        'ratio_g3_g0': rates[3] / rates[0] if rates[0] > 0 else None
    }
    print(f"{layer:<35} | Slope: {slope:+.3f} | R²: {r2:.3f} | Type: {scaling_type:<12} | g3/g0: {rates[3]/rates[0]:.2f}x")

# Categorize layers
inverse_layers = [l for l, r in scaling_results.items() if r['type'] == 'inverse']
sublinear_layers = [l for l, r in scaling_results.items() if r['type'] == 'sublinear']
linear_layers = [l for l, r in scaling_results.items() if r['type'] == 'linear']
superlinear_layers = [l for l, r in scaling_results.items() if r['type'] == 'superlinear']

print("\n" + "="*80)
print("LAYER CATEGORIZATION")
print("="*80)
print(f"\nINVERSE SCALING ({len(inverse_layers)} layers):")
for l in inverse_layers:
    print(f"  - {l}: g3/g0 = {scaling_results[l]['ratio_g3_g0']:.2f}x")
    
print(f"\nSUBLINEAR SCALING ({len(sublinear_layers)} layers):")
for l in sublinear_layers:
    print(f"  - {l}: g3/g0 = {scaling_results[l]['ratio_g3_g0']:.2f}x")
    
print(f"\nLINEAR SCALING ({len(linear_layers)} layers):")
for l in linear_layers:
    print(f"  - {l}: g3/g0 = {scaling_results[l]['ratio_g3_g0']:.2f}x")
    
print(f"\nSUPERLINEAR SCALING ({len(superlinear_layers)} layers):")
for l in superlinear_layers:
    print(f"  - {l}: g3/g0 = {scaling_results[l]['ratio_g3_g0']:.2f}x")

# ============= PLOT 1: Firing Rate Scaling by Layer Type =============
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Color scheme
colors = {
    'inverse': '#d62728',      # Red
    'sublinear': '#ff7f0e',    # Orange
    'linear': '#2ca02c',       # Green
    'superlinear': '#1f77b4',  # Blue
}

# Plot 1a: Inverse scaling layers
ax = axes[0, 0]
for layer in inverse_layers:
    rates = firing_rates[layer]
    short_name = layer.split('.')[-1] if len(layer.split('.')) > 2 else layer
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('INVERSE Scaling Layers\n(Higher firing rate at lower granularity)', fontsize=11, color=colors['inverse'])
ax.set_xticks(granularities)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 3.2)

# Plot 1b: Sublinear scaling layers
ax = axes[0, 1]
for layer in sublinear_layers:
    rates = firing_rates[layer]
    short_name = '.'.join(layer.split('.')[-2:]) if len(layer.split('.')) > 2 else layer
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('SUBLINEAR Scaling Layers\n(Firing rate increases slower than granularity)', fontsize=11, color=colors['sublinear'])
ax.set_xticks(granularities)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 3.2)

# Plot 1c: Linear scaling layers
ax = axes[1, 0]
for layer in linear_layers:
    rates = firing_rates[layer]
    short_name = '.'.join(layer.split('.')[-2:]) if len(layer.split('.')) > 2 else layer
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('LINEAR Scaling Layers\n(Firing rate proportional to granularity)', fontsize=11, color=colors['linear'])
ax.set_xticks(granularities)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 3.2)

# Plot 1d: Superlinear scaling layers
ax = axes[1, 1]
for layer in superlinear_layers:
    rates = firing_rates[layer]
    short_name = '.'.join(layer.split('.')[-2:]) if len(layer.split('.')) > 2 else layer
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('SUPERLINEAR Scaling Layers\n(Firing rate increases faster than granularity)', fontsize=11, color=colors['superlinear'])
ax.set_xticks(granularities)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 3.2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_by_category.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir}/scaling_by_category.png")
plt.close()

# ============= PLOT 2: Scaling Exponent Summary =============
fig, ax = plt.subplots(figsize=(14, 8))

layers = list(scaling_results.keys())
slopes = [scaling_results[l]['slope'] for l in layers]
layer_colors = [colors[scaling_results[l]['type']] for l in layers]

# Create short names
short_names = []
for l in layers:
    parts = l.split('.')
    if len(parts) >= 3:
        short_names.append(f"{parts[-2]}.{parts[-1]}")
    else:
        short_names.append(l)

x_pos = np.arange(len(layers))
bars = ax.bar(x_pos, slopes, color=layer_colors, edgecolor='black', linewidth=0.5)

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Linear (slope=1)')
ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Scaling Exponent (log-log slope)', fontsize=12)
ax.set_title('Firing Rate Scaling Exponent by Layer\n(Negative = Inverse, 0-0.5 = Sublinear, 0.5-1.5 = Linear, >1.5 = Superlinear)', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['inverse'], label='Inverse'),
    Patch(facecolor=colors['sublinear'], label='Sublinear'),
    Patch(facecolor=colors['linear'], label='Linear'),
    Patch(facecolor=colors['superlinear'], label='Superlinear'),
]
ax.legend(handles=legend_elements, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_exponents.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/scaling_exponents.png")
plt.close()

# ============= PLOT 3: Normalized Firing Rate (relative to g0) =============
fig, ax = plt.subplots(figsize=(12, 8))

for layer, rates in firing_rates.items():
    normalized = [r / rates[0] for r in rates]  # Normalize to g0
    scaling_type = scaling_results[layer]['type']
    color = colors[scaling_type]
    alpha = 0.8 if scaling_type in ['inverse', 'superlinear'] else 0.4
    linewidth = 2.5 if scaling_type in ['inverse', 'superlinear'] else 1.5
    
    short_name = '.'.join(layer.split('.')[-2:]) if len(layer.split('.')) > 2 else layer
    ax.plot(granularities, normalized, 'o-', color=color, alpha=alpha, 
            linewidth=linewidth, markersize=6, label=short_name if scaling_type in ['inverse', 'superlinear'] else '')

# Reference lines
ax.plot(granularities, [1, 2, 4, 8], 'k--', linewidth=2, label='Perfect 2x scaling')
ax.axhline(y=1, color='gray', linestyle=':', linewidth=1)

ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Normalized Firing Rate (relative to g0)', fontsize=12)
ax.set_title('Normalized Firing Rate Scaling\n(All layers normalized to their g0 value)', fontsize=12)
ax.set_xticks(granularities)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.set_ylim(0.1, 20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'normalized_scaling.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/normalized_scaling.png")
plt.close()

# ============= PLOT 4: Spike Count vs Parameters =============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Total spikes per granularity
total_spikes = [550280, 797112, 1185049, 1900198]

# Plot 4a: Total spikes vs parameters
ax = axes[0]
ax.plot(parameters, total_spikes, 'bo-', linewidth=2, markersize=10)
for i, (p, s) in enumerate(zip(parameters, total_spikes)):
    ax.annotate(f'g{i}', (p, s), textcoords="offset points", xytext=(10, 5), fontsize=10)

# Fit line
slope, intercept, r_value, _, _ = stats.linregress(np.log(parameters), np.log(total_spikes))
x_fit = np.linspace(min(parameters), max(parameters), 100)
y_fit = np.exp(intercept) * x_fit**slope
ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, label=f'Power fit: slope={slope:.2f}')

ax.set_xlabel('Parameters', fontsize=12)
ax.set_ylabel('Total Spikes per Inference', fontsize=12)
ax.set_title('Total Spike Count vs Model Size', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4b: Firing rate breakdown by layer type
ax = axes[1]

# Aggregate by type
type_spikes = {'inverse': [], 'sublinear': [], 'linear': [], 'superlinear': []}
for g in range(4):
    for t in type_spikes.keys():
        type_spikes[t].append(sum(spike_counts[l][g] for l in spike_counts if scaling_results[l]['type'] == t))

x = np.arange(4)
width = 0.2
for i, (t, spikes) in enumerate(type_spikes.items()):
    offset = (i - 1.5) * width
    ax.bar(x + offset, [s/1e6 for s in spikes], width, label=t.capitalize(), color=colors[t])

ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Spikes (Millions)', fontsize=12)
ax.set_title('Spike Distribution by Scaling Category', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['g0', 'g1', 'g2', 'g3'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spikes_vs_parameters.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/spikes_vs_parameters.png")
plt.close()

# ============= PLOT 5: Layer-wise Heatmap =============
fig, ax = plt.subplots(figsize=(10, 12))

# Create matrix of normalized values (g3/g0 ratio for each layer)
layer_names = list(firing_rates.keys())
ratios = np.array([[rates[g]/rates[0] for g in range(4)] for rates in firing_rates.values()])

# Sort by g3/g0 ratio
sort_idx = np.argsort(ratios[:, 3])
ratios_sorted = ratios[sort_idx]
names_sorted = [layer_names[i] for i in sort_idx]

# Short names
short_sorted = []
for l in names_sorted:
    parts = l.split('.')
    if len(parts) >= 3:
        short_sorted.append(f"{parts[-2]}.{parts[-1]}")
    else:
        short_sorted.append(l)

im = ax.imshow(ratios_sorted, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=10)

ax.set_xticks(range(4))
ax.set_xticklabels(['g0\n(baseline)', 'g1', 'g2', 'g3'])
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(short_sorted, fontsize=9)

# Add text annotations
for i in range(len(names_sorted)):
    for j in range(4):
        val = ratios_sorted[i, j]
        color = 'white' if val < 0.5 or val > 5 else 'black'
        ax.text(j, i, f'{val:.2f}x', ha='center', va='center', color=color, fontsize=8)

ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('Firing Rate Scaling Heatmap\n(Normalized to g0, sorted by g3/g0 ratio)', fontsize=12)

cbar = plt.colorbar(im, ax=ax, label='Relative Firing Rate (vs g0)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_heatmap.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/scaling_heatmap.png")
plt.close()

# ============= PLOT 6: Feature Extractor vs Transformer Analysis =============
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Categorize layers
patch_layers = [l for l in firing_rates if 'patch_embed' in l]
attn_layers = [l for l in firing_rates if 'attn' in l]
mlp_layers = [l for l in firing_rates if 'mlp' in l]

# Plot 6a: Patch Embedding
ax = axes[0]
for layer in patch_layers:
    rates = firing_rates[layer]
    short_name = layer.split('.')[-1]
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('Patch Embedding Layers\n(Feature Extraction)', fontsize=12)
ax.set_xticks(granularities)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 6b: Attention
ax = axes[1]
for layer in attn_layers:
    rates = firing_rates[layer]
    parts = layer.split('.')
    short_name = f"b{parts[1][-1]}.{parts[-1]}"
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('Attention Layers\n(Self-Attention Mechanism)', fontsize=12)
ax.set_xticks(granularities)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 6c: MLP
ax = axes[2]
for layer in mlp_layers:
    rates = firing_rates[layer]
    parts = layer.split('.')
    short_name = f"b{parts[1][-1]}.{parts[-1]}"
    ax.plot(granularities, [r*100 for r in rates], 'o-', label=short_name, linewidth=2, markersize=8)
ax.set_xlabel('Granularity', fontsize=12)
ax.set_ylabel('Firing Rate (%)', fontsize=12)
ax.set_title('MLP Layers\n(Feed-Forward Network)', fontsize=12)
ax.set_xticks(granularities)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'component_analysis.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/component_analysis.png")
plt.close()

# ============= Summary Statistics =============
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBy Layer Category:")
for category, layer_list in [('Patch Embed', patch_layers), ('Attention', attn_layers), ('MLP', mlp_layers)]:
    avg_slope = np.mean([scaling_results[l]['slope'] for l in layer_list])
    avg_ratio = np.mean([scaling_results[l]['ratio_g3_g0'] for l in layer_list])
    print(f"  {category:15} | Avg Slope: {avg_slope:+.3f} | Avg g3/g0 ratio: {avg_ratio:.2f}x")

print("\nBy Scaling Type:")
for stype in ['inverse', 'sublinear', 'linear', 'superlinear']:
    layers_of_type = [l for l in scaling_results if scaling_results[l]['type'] == stype]
    if layers_of_type:
        avg_ratio = np.mean([scaling_results[l]['ratio_g3_g0'] for l in layers_of_type])
        print(f"  {stype:15} | Count: {len(layers_of_type):2} | Avg g3/g0 ratio: {avg_ratio:.2f}x")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
