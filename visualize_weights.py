#!/usr/bin/env python3
"""
Weight Visualization Script for ActorCritic Network
Loads a checkpoint and visualizes layer weights as heatmaps grouped by module.

Usage:
    python visualize_weights.py --checkpoint outputs/RightCorlAllegroHandHora/test/stage1_nn/best.pth
    python visualize_weights.py --checkpoint outputs/RightCorlAllegroHandHora/test/stage2_nn/best.pth --output_dir weight_vis
"""

import argparse
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


def get_stage_prefix(checkpoint_path):
    """Extract stage prefix from checkpoint path."""
    if 'stage1' in checkpoint_path:
        return 's1_'
    elif 'stage2' in checkpoint_path:
        return 's2_'
    return ''


def visualize_mlp_flow(weights_list, biases_list, layer_names, module_name, output_dir, stage_prefix='', show_values=True, max_display_size=30):
    """
    Visualize MLP layers vertically: input (top) -> layer1 -> layer2 -> ... -> output (bottom)

    Args:
        weights_list: List of weight matrices
        biases_list: List of bias vectors
        layer_names: List of layer names
        module_name: Name of the module (e.g., 'actor_mlp', 'env_mlp')
        output_dir: Directory to save the figure
        stage_prefix: Prefix for filename (e.g., 's1_' or 's2_')
        show_values: Whether to show actual weight values in cells
        max_display_size: Maximum size to show values (for readability)
    """
    n_layers = len(weights_list)

    # Calculate max width for consistent subplot widths
    max_width = max(w.shape[0] for w in weights_list)  # max output dim (will be x-axis)

    # Create figure with subplots stacked vertically
    fig, axes = plt.subplots(n_layers, 1, figsize=(max(12, max_width * 0.1), 3 * n_layers + 1))
    if n_layers == 1:
        axes = [axes]

    # Find global min/max for consistent colormap
    all_weights = np.concatenate([w.flatten() for w in weights_list])
    vmax = max(abs(all_weights.min()), abs(all_weights.max()))
    vmin = -vmax
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for idx, (weight, bias, name, ax) in enumerate(zip(weights_list, biases_list, layer_names, axes)):
        # Weight shape: (output_dim, input_dim)
        # We want: y-axis = input (top to bottom), x-axis = output
        # So we use weight.T which gives (input_dim, output_dim)
        # But we want input on y-axis going down, so no additional flip needed
        weight_display = weight  # (output_dim, input_dim) -> display as is, x=output, y=input
        h, w = weight_display.shape  # h=output_dim, w=input_dim

        im = ax.imshow(weight_display.T, cmap=cmap, norm=norm, aspect='auto')
        # weight_display.T -> (input_dim, output_dim), so y=input, x=output

        # Show values if small enough
        in_dim, out_dim = weight_display.T.shape
        if show_values and in_dim <= max_display_size and out_dim <= max_display_size:
            fontsize = max(4, min(7, 150 // max(in_dim, out_dim)))
            for i in range(in_dim):
                for j in range(out_dim):
                    val = weight_display.T[i, j]
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=fontsize, color=text_color)

        # Labels - input on y-axis (top), output on x-axis (bottom)
        ax.set_ylabel(f'Input ({in_dim})', fontsize=9)
        ax.set_xlabel(f'Output ({out_dim})', fontsize=9)
        ax.set_title(f'{name}: {in_dim} → {out_dim}', fontsize=10, fontweight='bold')

        # Ticks
        if out_dim <= 32:
            ax.set_xticks(range(out_dim))
        if in_dim <= 32:
            ax.set_yticks(range(in_dim))

    # Title with architecture info
    dims = " → ".join([str(weights_list[0].shape[1])] + [str(w.shape[0]) for w in weights_list])
    fig.suptitle(f'{module_name}  |  Architecture: {dims}', fontsize=14, fontweight='bold')

    plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.3)

    # Add horizontal colorbar at top right
    cbar_ax = fig.add_axes([0.75, 0.94, 0.2, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weight Value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    safe_name = module_name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'{stage_prefix}{safe_name}_flow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def visualize_conv1d_flow(conv_weights, conv_names, module_name, output_dir, stage_prefix=''):
    """
    Visualize Conv1d layers in a flow diagram.
    """
    n_layers = len(conv_weights)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 8))
    if n_layers == 1:
        axes = [axes]

    # Find global min/max
    all_weights = np.concatenate([w.flatten() for w in conv_weights])
    vmax = max(abs(all_weights.min()), abs(all_weights.max()))
    vmin = -vmax
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for idx, (weight, name, ax) in enumerate(zip(conv_weights, conv_names, axes)):
        out_ch, in_ch, k_size = weight.shape

        # Reshape to 2D: (out_ch, in_ch * k_size)
        weight_2d = weight.reshape(out_ch, -1).T  # (in_ch * k_size, out_ch)

        im = ax.imshow(weight_2d, cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(f'{name}\n({in_ch}×{k_size} → {out_ch})', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'Output Channels ({out_ch})', fontsize=9)
        ax.set_ylabel(f'Input (Ch×Kernel = {in_ch}×{k_size})', fontsize=9)

        # Add horizontal lines to separate input channels
        for i in range(1, in_ch):
            ax.axhline(y=i * k_size - 0.5, color='gray', linewidth=0.5, linestyle='--')

    fig.suptitle(f'{module_name} - Temporal Convolutions', fontsize=14, fontweight='bold')

    plt.subplots_adjust(top=0.90, bottom=0.08)

    # Add horizontal colorbar at top right
    cbar_ax = fig.add_axes([0.75, 0.94, 0.2, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weight Value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    safe_name = module_name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'{stage_prefix}{safe_name}_conv_flow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def visualize_single_layer(weight, bias, name, output_dir, stage_prefix='', show_values=True, max_display_size=30):
    """Visualize a single layer (for output heads like value) - horizontal layout: input (top) -> output (bottom)."""
    if weight.ndim == 0:
        print(f"  Skipping {name} (scalar)")
        return

    if weight.ndim == 1:
        weight = weight.reshape(1, -1)

    # weight shape: (output_dim, input_dim)
    # For horizontal layout with top-to-bottom = input-to-output:
    # y-axis = input_dim (top to bottom), x-axis = output_dim
    weight_display = weight.T  # (input_dim, output_dim)
    in_dim, out_dim = weight_display.shape

    # Horizontal layout: width based on output dim, height based on input dim
    # Make it wide (horizontal) with input going down
    fig_width = max(10, out_dim * 2)
    fig_height = max(8, in_dim * 0.08)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = max(abs(weight_display.min()), abs(weight_display.max()))
    vmin = -vmax
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.imshow(weight_display, cmap=cmap, norm=norm, aspect='auto')

    if show_values and in_dim <= max_display_size and out_dim <= max_display_size:
        fontsize = max(5, min(8, 180 // max(in_dim, out_dim)))
        for i in range(in_dim):
            for j in range(out_dim):
                val = weight_display[i, j]
                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=fontsize, color=text_color)

    ax.set_ylabel(f'Input ({in_dim})', fontsize=10)
    ax.set_xlabel(f'Output ({out_dim})', fontsize=10)

    if out_dim <= 20:
        ax.set_xticks(range(out_dim))
    if in_dim <= 32:
        ax.set_yticks(range(in_dim))

    # Title
    fig.suptitle(f'{name}: {in_dim} → {out_dim}', fontsize=12, fontweight='bold')

    plt.subplots_adjust(top=0.88, bottom=0.1)

    # Add horizontal colorbar at top right
    cbar_ax = fig.add_axes([0.75, 0.92, 0.2, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weight', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    safe_name = name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(output_dir, f'{stage_prefix}{safe_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def visualize_mu_sigma(mu_weight, mu_bias, sigma, output_dir, stage_prefix=''):
    """Visualize mu weight and sigma together in one figure with aligned x-axis."""

    # Transpose mu for input -> output flow
    mu_t = mu_weight.T  # (input_dim, output_dim) = (128, 16)
    in_dim, out_dim = mu_t.shape

    # Create figure with shared x-axis
    fig, axes = plt.subplots(2, 1, figsize=(max(14, out_dim * 0.8), in_dim * 0.1 + 4),
                              gridspec_kw={'height_ratios': [in_dim // 4, 1]},
                              sharex=True)

    # Mu weight heatmap
    ax_mu = axes[0]
    vmax_mu = max(abs(mu_t.min()), abs(mu_t.max()))
    vmin_mu = -vmax_mu
    cmap = plt.cm.RdBu_r
    norm_mu = mcolors.TwoSlopeNorm(vmin=vmin_mu, vcenter=0, vmax=vmax_mu)

    im_mu = ax_mu.imshow(mu_t, cmap=cmap, norm=norm_mu, aspect='auto')
    ax_mu.set_ylabel(f'Input ({in_dim})', fontsize=10)
    ax_mu.set_title(f'mu.weight: {in_dim} → {out_dim}', fontsize=11, fontweight='bold')

    # Sigma bar chart
    ax_sigma = axes[1]
    x = np.arange(len(sigma))
    colors = ['green' if s >= 0 else 'red' for s in sigma]
    bars = ax_sigma.bar(x, sigma, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, width=0.8)

    # Calculate y-axis range for proper value placement
    y_max = max(sigma.max(), 0) + 0.15
    y_min = min(sigma.min(), 0) - 0.05
    ax_sigma.set_ylim(y_min, y_max)

    # Add value labels above bars (at fixed height above all bars)
    label_y = y_max - 0.03  # Position near top of plot
    for i, val in enumerate(sigma):
        ax_sigma.text(i, label_y, f'{val:.3f}', ha='center', va='top',
                     fontsize=9, fontweight='bold', color='darkblue')

    ax_sigma.set_ylabel('Sigma', fontsize=10)
    ax_sigma.set_xlabel('Action Index', fontsize=10)
    ax_sigma.set_title(f'sigma (learnable std)', fontsize=11, fontweight='bold')
    ax_sigma.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_sigma.set_xlim(-0.5, len(sigma) - 0.5)
    ax_sigma.set_xticks(range(len(sigma)))

    # Main title
    fig.suptitle('Action Output Head (mu & sigma)', fontsize=14, fontweight='bold')

    plt.subplots_adjust(top=0.88, bottom=0.1, hspace=0.3)

    # Add horizontal colorbar at top right for mu
    cbar_ax = fig.add_axes([0.75, 0.92, 0.2, 0.02])
    cbar = fig.colorbar(im_mu, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('mu Weight', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    save_path = os.path.join(output_dir, f'{stage_prefix}mu_sigma.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def visualize_checkpoint(checkpoint_path, output_dir):
    """Load checkpoint and visualize weights grouped by module."""

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    os.makedirs(output_dir, exist_ok=True)

    # Get stage prefix from checkpoint path
    stage_prefix = get_stage_prefix(checkpoint_path)
    stage_name = "Stage 1" if stage_prefix == 's1_' else "Stage 2" if stage_prefix == 's2_' else "Unknown"
    print(f"Detected: {stage_name}")

    # Get model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"\nFound {len(state_dict)} parameters in checkpoint")
    print("\n" + "="*60)
    print("Layer Summary:")
    print("="*60)

    for name, param in state_dict.items():
        shape_str = str(list(param.shape))
        print(f"  {name}: {shape_str}")

    # Group parameters by module
    modules = defaultdict(lambda: {'weights': [], 'biases': [], 'names': []})
    single_layers = {}
    conv_modules = defaultdict(lambda: {'weights': [], 'names': []})

    for name, param in state_dict.items():
        weight = param.numpy()

        # Parse module name
        parts = name.split('.')

        if 'mlp' in name and 'weight' in name:
            # MLP layers: actor_mlp.mlp.0.weight -> actor_mlp
            module_name = parts[0]
            layer_idx = int(parts[2])
            modules[module_name]['weights'].append((layer_idx, weight))
            modules[module_name]['names'].append(f'Layer {layer_idx // 2}')

        elif 'mlp' in name and 'bias' in name:
            module_name = parts[0]
            layer_idx = int(parts[2])
            modules[module_name]['biases'].append((layer_idx, weight))

        elif 'adapt_tconv' in name:
            # Conv layers in adaptation module
            if 'temporal_aggregation' in name and 'weight' in name:
                layer_idx = int(parts[2])
                conv_modules['adapt_tconv.temporal']['weights'].append((layer_idx, weight))
                conv_modules['adapt_tconv.temporal']['names'].append(f'Conv {layer_idx // 2}')
            elif 'channel_transform' in name and 'weight' in name:
                layer_idx = int(parts[2])
                conv_modules['adapt_tconv.channel']['weights'].append((layer_idx, weight))
                conv_modules['adapt_tconv.channel']['names'].append(f'Linear {layer_idx // 2}')
            elif 'low_dim_proj' in name and 'weight' in name:
                single_layers['adapt_tconv.low_dim_proj'] = (weight, None)

        elif name in ['mu.weight', 'value.weight', 'sigma']:
            bias_name = name.replace('weight', 'bias')
            bias = state_dict.get(bias_name, None)
            bias = bias.numpy() if bias is not None else None
            single_layers[name.replace('.weight', '')] = (weight, bias)

    print("\n" + "="*60)
    print("Visualizing weights (grouped by module)...")
    print("="*60)

    # Visualize MLP modules
    for module_name, data in modules.items():
        if data['weights']:
            # Sort by layer index
            weights_sorted = sorted(data['weights'], key=lambda x: x[0])
            biases_sorted = sorted(data['biases'], key=lambda x: x[0])

            weights = [w for _, w in weights_sorted]
            biases = [b for _, b in biases_sorted]
            names = [f'Layer {i}' for i in range(len(weights))]

            visualize_mlp_flow(weights, biases, names, module_name, output_dir, stage_prefix)

    # Visualize Conv modules
    for module_name, data in conv_modules.items():
        if data['weights']:
            weights_sorted = sorted(data['weights'], key=lambda x: x[0])
            weights = [w for _, w in weights_sorted]
            names = [n for _, n in sorted(zip([x[0] for x in data['weights']], data['names']))]

            if weights[0].ndim == 3:  # Conv1d
                visualize_conv1d_flow(weights, names, module_name, output_dir, stage_prefix)
            else:  # Linear in channel_transform
                biases = [np.zeros(w.shape[0]) for w in weights]  # dummy biases
                visualize_mlp_flow(weights, biases, names, module_name, output_dir, stage_prefix)

    # Visualize single layers (output heads)
    for name, (weight, bias) in single_layers.items():
        if name not in ['mu', 'sigma']:  # Skip mu and sigma here, visualize together below
            visualize_single_layer(weight, bias, name, output_dir, stage_prefix)

    # Visualize mu and sigma together
    if 'mu.weight' in state_dict and 'sigma' in state_dict:
        mu_weight = state_dict['mu.weight'].numpy()
        mu_bias = state_dict.get('mu.bias', None)
        mu_bias = mu_bias.numpy() if mu_bias is not None else None
        sigma = state_dict['sigma'].numpy()
        visualize_mu_sigma(mu_weight, mu_bias, sigma, output_dir, stage_prefix)

    print(f"\nAll visualizations saved to: {output_dir}")

    # Create summary
    create_summary(state_dict, output_dir, stage_prefix)


def create_summary(state_dict, output_dir, stage_prefix=''):
    """Create a summary figure showing weight statistics."""

    stats = []
    display_names = []

    for name, param in state_dict.items():
        weight = param.numpy()
        if weight.size == 0:
            continue
        stats.append({
            'name': name,
            'shape': weight.shape,
            'mean': weight.mean(),
            'std': weight.std(),
            'min': weight.min(),
            'max': weight.max(),
            'num_params': weight.size
        })
        # Create descriptive display name
        # e.g., "actor_mlp.mlp.0.weight" -> "actor_mlp.L0.W"
        #       "env_mlp.mlp.2.bias" -> "env_mlp.L1.B"
        parts = name.split('.')
        if len(parts) >= 4 and 'mlp' in name:
            module = parts[0]
            layer_num = int(parts[2]) // 2
            param_type = 'W' if 'weight' in name else 'B'
            display_name = f"{module}.L{layer_num}.{param_type}"
        elif 'adapt_tconv' in name:
            if 'temporal' in name:
                layer_num = int(parts[2]) // 2
                param_type = 'W' if 'weight' in name else 'B'
                display_name = f"tconv.T{layer_num}.{param_type}"
            elif 'channel' in name:
                layer_num = int(parts[2]) // 2
                param_type = 'W' if 'weight' in name else 'B'
                display_name = f"tconv.C{layer_num}.{param_type}"
            elif 'low_dim' in name:
                param_type = 'W' if 'weight' in name else 'B'
                display_name = f"tconv.proj.{param_type}"
            else:
                display_name = name
        elif name in ['mu.weight', 'mu.bias']:
            display_name = 'mu.W' if 'weight' in name else 'mu.B'
        elif name in ['value.weight', 'value.bias']:
            display_name = 'value.W' if 'weight' in name else 'value.B'
        elif name == 'sigma':
            display_name = 'sigma'
        else:
            display_name = name[:15] if len(name) > 15 else name

        display_names.append(display_name)

    if not stats:
        return

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    x = range(len(stats))

    # Mean values
    ax = axes[0, 0]
    means = [s['mean'] for s in stats]
    colors = ['red' if m < 0 else 'blue' for m in means]
    ax.bar(x, means, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=60, ha='right', fontsize=7)
    ax.set_title('Mean Weight Values', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean')
    ax.set_xlabel('Layer (Module.LayerNum.Type: W=Weight, B=Bias)')

    # Std values
    ax = axes[0, 1]
    stds = [s['std'] for s in stats]
    ax.bar(x, stds, color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=60, ha='right', fontsize=7)
    ax.set_title('Weight Standard Deviations', fontweight='bold')
    ax.set_ylabel('Std')
    ax.set_xlabel('Layer (Module.LayerNum.Type: W=Weight, B=Bias)')

    # Min/Max values
    ax = axes[1, 0]
    mins = [s['min'] for s in stats]
    maxs = [s['max'] for s in stats]
    ax.bar(x, maxs, color='blue', alpha=0.7, label='Max')
    ax.bar(x, mins, color='red', alpha=0.7, label='Min')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=60, ha='right', fontsize=7)
    ax.set_title('Weight Min/Max Values', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Layer (Module.LayerNum.Type: W=Weight, B=Bias)')

    # Number of parameters
    ax = axes[1, 1]
    num_params = [s['num_params'] for s in stats]
    ax.bar(x, num_params, color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=60, ha='right', fontsize=7)
    ax.set_title('Number of Parameters per Layer', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Layer (Module.LayerNum.Type: W=Weight, B=Bias)')
    ax.set_yscale('log')

    total_params = sum(num_params)
    stage_name = "Stage 1" if stage_prefix == 's1_' else "Stage 2" if stage_prefix == 's2_' else ""
    fig.suptitle(f'{stage_name} Weight Statistics Summary  |  Total Parameters: {total_params:,}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{stage_prefix}summary_statistics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize network weights from checkpoint')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to checkpoint file (.pth)')
    parser.add_argument('--output_dir', '-o', type=str, default='weight_visualizations',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    visualize_checkpoint(args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
