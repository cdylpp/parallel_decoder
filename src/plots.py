import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame, shots: int = 2000):
    """
    Enhanced plot showing both decoder comparison and their relative differences.
    
    Creates a 2-row plot:
    - Top: Traditional log-scale error rate comparison
    - Bottom: Relative error difference highlighting subtle variations
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1], sharex=ax_main)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data['lattice_size'].unique())))
    
    # Main comparison plot
    for idx, size in enumerate(sorted(data['lattice_size'].unique())):
        global_subset = data[(data['lattice_size'] == size) & (data['type'] == 'global')]
        window_subset = data[(data['lattice_size'] == size) & (data['type'] == 'parallel')]
        
        # Error bars based on binomial standard error
        global_err = np.sqrt(global_subset['logical_error_rate'] * 
                            (1 - global_subset['logical_error_rate']) / shots)
        window_err = np.sqrt(window_subset['logical_error_rate'] * 
                            (1 - window_subset['logical_error_rate']) / shots)
        
        # Main plot with error bars
        ax_main.fill_between(
            global_subset['rounds'],
            -global_err,
            +global_err,
            alpha=0.5,
            linewidth=0
        )

        ax_main.plot(
            global_subset['rounds'],
            global_subset['logical_error_rate'],
            linestyle='-',
            color=colors[idx],
            label=f'Global L={size}',
            markersize=6,
            capsize=3,
            alpha=0.8
        )

        
        
        ax_main.errorbar(
            window_subset['rounds'],
            window_subset['logical_error_rate'],
            yerr=window_err,
            marker='s',
            linestyle='--',
            color=colors[idx],
            label=f'Parallel L={size}',
            markersize=7,
            markeredgewidth=1.5,
            capsize=3,
            alpha=0.8
        )
        
        # Difference plot: (Sliding - Global) / Global * 100 for percentage
        # Align data by rounds
        merged = pd.merge(
            global_subset[['rounds', 'logical_error_rate']],
            window_subset[['rounds', 'logical_error_rate']],
            on='rounds',
            suffixes=('_global', '_parallel')
        )
        
        # Relative percentage difference
        rel_diff = ((merged['logical_error_rate_parallel'] - merged['logical_error_rate_global']) 
                    / merged['logical_error_rate_global'] * 100)
        
        ax_diff.plot(
            merged['rounds'],
            rel_diff,
            marker='D',
            linestyle='-',
            color=colors[idx],
            label=f'L={size}',
            markersize=5,
            linewidth=2,
            alpha=0.8
        )
    
    # Main plot formatting
    ax_main.set_ylabel('Logical Error Rate', fontsize=13, fontweight='bold')
    ax_main.set_ylim(1e-4, 1e-0)
    ax_main.set_xlim(0, 230)
    ax_main.set_yscale('log')
    ax_main.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.7)
    ax_main.set_title('Logical Error Rate: Global vs Parallel Window Decoder', 
                     fontsize=14, fontweight='bold', pad=15)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    # Difference plot formatting
    ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_diff.set_xlabel('Rounds', fontsize=13, fontweight='bold')
    ax_diff.set_ylabel('Relative Error\nDifference (%)', fontsize=11, fontweight='bold')
    ax_diff.set_xlim(0, 230)
    ax_diff.legend(loc='best', fontsize=9, ncol=4, framealpha=0.9)
    ax_diff.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax_diff.set_title('Relative Difference: (Parallel - Global) / Global × 100%', 
                     fontsize=11, style='italic', pad=10)
    
    # Add subtle background shading for positive/negative regions
    ax_diff.axhspan(0, ax_diff.get_ylim()[1], alpha=0.1, color='red', zorder=-1)
    ax_diff.axhspan(ax_diff.get_ylim()[0], 0, alpha=0.1, color='green', zorder=-1)
    
    # plt.tight_layout()
    plt.show()
    return
    