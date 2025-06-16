import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import matplotlib.font_manager
mpl.use('Agg')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Configure plotting parameters with reduced heights
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times',
    'font.size': 20,  # Slightly smaller font
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.figsize': (6, 2.5),  # Reduced height (2.5 inches)
    'figure.dpi': 600,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'axes.edgecolor': '0.3',
    'axes.linewidth': 0.8,
    'legend.fontsize': 20,  # Smaller legend
    'text.usetex': True, # time news roman
    'mathtext.fontset': 'custom'
})

# Use consistent colors
# dark blue: (31,120,180)
LINE_COLOR = '#1f77b4'  # Dark blue color for lines  
HIGHLIGHT_COLOR = '#d62728'
BASELINE_COLOR = '#7f7f7f'  # Gray color for baseline
# LABEL_COLOR = '#666666'  # New color for labels
LABEL_COLOR = 'black'  # Black color for labels


# size
marker_size=4
line_width=1.8
value_fontsize=18

def create_ccm_layer_plot():
    """Create plot for CCM layer experiment"""
    layers = [0, 1, 2, 3, 4, 5, 6]
    avg_maps = [62.1, 63.3, 63.0, 64.3, 62.1, 63.1, 62.3]
    baseline_value = avg_maps[0]  # Value when layer=0
    
    fig, ax = plt.subplots()
    
    # Draw baseline (dashed line for layer=0)
    ax.axhline(y=baseline_value, color=BASELINE_COLOR, linestyle='--', 
               linewidth=1.2, alpha=0.7)
    
    # Plot the main data line
    ax.plot(layers, avg_maps, 'o-', color=LINE_COLOR, linewidth=line_width, 
            markersize=marker_size, zorder=3)
    
    # Highlight best result with star
    best_idx = np.argmax(avg_maps)
    ax.plot(layers[best_idx], avg_maps[best_idx], '*', markersize=9, 
            color=HIGHLIGHT_COLOR, markeredgewidth=1.2, zorder=4)
    
    # Add baseline annotation
    # ax.text(6.1, baseline_value, 'Baseline (0 layers)', 
    #         fontsize=8, color=BASELINE_COLOR, va='center')
    
    # Add value labels
    for x, y in zip(layers, avg_maps):
        ax.text(x, y+0.1, f'{y:.1f}', color=LABEL_COLOR, ha='center', va='bottom', fontsize=value_fontsize)
    
    # Axis formatting
    # ax.set_xlabel('Number of CCM Layers', fontweight='bold')
    # ax.set_ylabel('Avg. mAP (\%)', fontweight='bold')  # Shortened label
    ax.set_xticks(layers)
    ax.set_xlim(-0.3, 6.3)
    ax.set_ylim(61.5, 65)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    plt.tight_layout()
    plt.savefig('ablation/ccm_layers_effect.pdf', bbox_inches='tight')
    plt.close()

def create_omega_plot():
    """Create plot for fusion weight experiment"""
    omegas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    avg_maps = [54.9, 58.7, 61.2, 62.6, 63.2, 63.9, 64.3, 64.3, 63.6, 63.2, 62.4]
    baseline_branches = [52.7, 61.7]
    branch_names = ['Skeleton', 'Appearance']
    
    
    fig, ax = plt.subplots()
    ax.plot(omegas, avg_maps, 'o-', color=LINE_COLOR, linewidth=line_width, markersize=marker_size)
    
    # Highlight best result with star
    best_idx = 6  # ω=0.6 has highest overall mAP
    ax.plot(omegas[best_idx], avg_maps[best_idx], '*', markersize=9, 
            color=HIGHLIGHT_COLOR, markeredgewidth=1.2)
    
    # Add value labels
    for x, y in zip(omegas, avg_maps):
        ax.text(x, y+0.3, f'{y:.1f}', color=LABEL_COLOR, ha='center', va='bottom', fontsize=value_fontsize)
    
    # Add baseline branches
    for i, baseline in enumerate(baseline_branches):
        ax.axhline(y=baseline, color=BASELINE_COLOR, linestyle='--',
                   linewidth=1.2, alpha=0.7, zorder=2)
    
    
    ax.text(1.03, baseline_branches[0], f'Skeleton Branch',
                fontsize=value_fontsize, color=LABEL_COLOR, va='bottom', ha='right', zorder=2) 
    # ax.text(0.20, baseline_branches[0], f'Skeleton Branch',
    #             fontsize=12, color=BASELINE_COLOR, va='top', ha='right', zorder=2) 
    ax.text(1.03, baseline_branches[1], f'Appearance Branch',
                fontsize=value_fontsize, color=LABEL_COLOR, va='top', ha='right', zorder=2) 

    # Axis formatting
    # ax.set_xlabel('Fusion Weight ($\omega$)', fontweight='bold')
    # ax.set_ylabel('Avg. mAP (\%)', fontweight='bold')  # Shortened label
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(51, 68)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig('ablation/fusion_weight_effect.pdf', bbox_inches='tight')
    plt.close()

def create_sigma_plot():
    """Create plot for standard deviation experiment"""
    sigmas = [1, 2, 4, 6, 8]
    avg_maps = [45.4, 47.5, 52.7, 52.1, 52.5]
    
    fig, ax = plt.subplots()
    ax.plot(sigmas, avg_maps, 'o-', color=LINE_COLOR, linewidth=line_width, markersize=marker_size)
    
    # Highlight best result with star
    best_idx = 2  # σ=4 has highest mAP
    ax.plot(sigmas[best_idx], avg_maps[best_idx], '*', markersize=9, 
            color=HIGHLIGHT_COLOR, markeredgewidth=1.2)
    
    # Add value labels
    for x, y in zip(sigmas, avg_maps):
        ax.text(x, y+0.3, f'{y:.1f}', color=LABEL_COLOR, ha='center', va='bottom', fontsize=value_fontsize)
    
    # Axis formatting
    # ax.set_xlabel('Standard Deviation ($\sigma$)', fontweight='bold')
    # ax.set_ylabel('Avg. mAP (\%)', fontweight='bold')  # Shortened label
    ax.set_xticks(sigmas)
    ax.set_ylim(44, 56)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig('ablation/std_dev_effect.pdf', bbox_inches='tight')
    plt.close()

# Generate all figures
create_ccm_layer_plot()
create_omega_plot()
create_sigma_plot()