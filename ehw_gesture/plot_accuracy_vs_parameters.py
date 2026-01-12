#!/usr/bin/env python3
"""
Plot accuracy vs parameters for different configurations.
Marker size is proportional to mean spikes per timestep.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('spike_analysis_results_a10/full_evaluation_results.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
configurations = []
parameters = []
accuracies = []
mean_spikes = []

for config_name, config_data in data.items():
    configurations.append(config_name)
    parameters.append(config_data['parameters'])
    accuracies.append(config_data['acc1'])
    mean_spikes.append(config_data['mean_spikes_per_timestep'])

# Convert to numpy arrays
parameters = np.array(parameters)
accuracies = np.array(accuracies)
mean_spikes = np.array(mean_spikes)

# Normalize spike counts for marker sizes (scale to reasonable range)
# Map spikes to marker sizes between 20 and 300
min_spike = mean_spikes.min()
max_spike = mean_spikes.max()
marker_sizes = 20 + (mean_spikes - min_spike) / (max_spike - min_spike) * 280

# Create the plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(parameters / 1e6, accuracies, s=marker_sizes, 
                     alpha=0.6, c=mean_spikes, cmap='viridis', edgecolors='black', linewidth=0.5)

# Add colorbar to show spike counts
cbar = plt.colorbar(scatter, label='Mean Spikes per Timestep')

# Labels and title
plt.xlabel('Parameters (Millions)', fontsize=12)
plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
plt.title('Accuracy vs Parameters (Marker Size ∝ Mean Spikes per Timestep)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend for marker sizes
# Create dummy scatter plots for legend
spike_examples = [min_spike, (min_spike + max_spike) / 2, max_spike]
size_examples = [20, 150, 300]
legend_elements = []
for spike_val, size_val in zip(spike_examples, size_examples):
    legend_elements.append(plt.scatter([], [], s=size_val, c='gray', alpha=0.6, 
                                      edgecolors='black', linewidth=0.5,
                                      label=f'{spike_val:.0f} spikes'))

plt.legend(handles=legend_elements, title='Mean Spikes/Timestep', 
          loc='lower right', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
plt.savefig('accuracy_vs_parameters.pdf', bbox_inches='tight')
print(f"Plots saved: accuracy_vs_parameters.png and accuracy_vs_parameters.pdf")

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Number of configurations: {len(configurations)}")
print(f"Parameter range: {parameters.min()/1e6:.2f}M - {parameters.max()/1e6:.2f}M")
print(f"Accuracy range: {accuracies.min():.2f}% - {accuracies.max():.2f}%")
print(f"Mean spikes range: {mean_spikes.min():.0f} - {mean_spikes.max():.0f}")

# Find best configurations
best_acc_idx = np.argmax(accuracies)
print(f"\nBest accuracy: {accuracies[best_acc_idx]:.2f}% with {parameters[best_acc_idx]/1e6:.2f}M parameters")
print(f"  Configuration: {configurations[best_acc_idx]}")
print(f"  Mean spikes/timestep: {mean_spikes[best_acc_idx]:.0f}")

# Find most efficient (highest acc/param ratio)
efficiency = accuracies / (parameters / 1e6)
most_efficient_idx = np.argmax(efficiency)
print(f"\nMost efficient: {efficiency[most_efficient_idx]:.2f} acc%/M params")
print(f"  Configuration: {configurations[most_efficient_idx]}")
print(f"  Accuracy: {accuracies[most_efficient_idx]:.2f}%, Parameters: {parameters[most_efficient_idx]/1e6:.2f}M")
print(f"  Mean spikes/timestep: {mean_spikes[most_efficient_idx]:.0f}")

plt.show()
