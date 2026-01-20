#!/usr/bin/env python3

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Convert 21cm to inches for matplotlib (21 cm / 2.54 cm/inch = 8.27 inches)
PLOT_WIDTH = 21 / 2.54  # 8.27 inches
PLOT_HEIGHT = 4 / 2.54  # 1.57 inches

def parse_roofline_results(filename):
    """Parse the roofline results file."""
    
    roofline_results = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Parse roofline results
        # Format: 512 x 512 x 512 ROOFLINE_MODEL 12109.74 GFLOPS ...
        match = re.search(r'(\d+)\s+x\s+(\d+)\s+x\s+(\d+)\s+ROOFLINE_MODEL\s+(\d+(?:\.\d+)?)\s+GFLOPS', line)
        if match:
            M = int(match.group(1))
            N = int(match.group(2))
            K = int(match.group(3))
            gflops = float(match.group(4))
            roofline_results[(M, N, K)] = gflops
    
    return roofline_results


def parse_benchmark_results(filename):
    """Parse the benchmark results file."""
    
    onednn_results = {}
    custom_gemm_results = defaultdict(list)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse oneDNN results
        if line.startswith("Running oneDNN:"):
            match = re.search(r'M=(\d+) N=(\d+) K=(\d+)', line)
            if match:
                M, N, K = int(match.group(1)), int(match.group(2)), int(match.group(3))
                # Look for performance in next few lines (increased to 100 to handle verbose output)
                for j in range(i+1, min(i+100, len(lines))):
                    perf_match = re.search(r'Performance:\s+(\d+(?:\.\d+)?)\s+GFLOPS', lines[j], re.IGNORECASE)
                    if perf_match:
                        gflops = float(perf_match.group(1))
                        onednn_results[(M, N, K)] = gflops
                        break
        
        # Parse custom GEMM results
        elif line.startswith("Running Custom GEMM:"):
            match = re.search(r'M=(\d+) N=(\d+) K=(\d+) BFK=(\d+) C=(\d+) ACT=(\d+)', line)
            if match:
                M = int(match.group(1))
                N = int(match.group(2))
                K = int(match.group(3))
                BFK = int(match.group(4))
                C = int(match.group(5))
                ACT = int(match.group(6))
                
                # Look for performance in next few lines (increased to 100 to handle verbose output)
                for j in range(i+1, min(i+100, len(lines))):
                    perf_match = re.search(r'Time is [0-9.]+\s+ms\s+\((\d+(?:\.\d+)?)\s+GFLOPS\)', lines[j], re.IGNORECASE)
                    if perf_match:
                        gflops = float(perf_match.group(1))
                        custom_gemm_results[(M, N, K)].append({
                            'BFK': BFK,
                            'C': C,
                            'ACT': ACT,
                            'gflops': gflops
                        })
                        break
        
        i += 1
    
    return onednn_results, custom_gemm_results


def find_max_performance(results, act_filter=None):
    """Find maximum performance for given ACT filter."""
    
    filtered = results
    if act_filter is not None:
        if callable(act_filter):
            filtered = [r for r in results if act_filter(r['ACT'])]
        else:
            filtered = [r for r in results if r['ACT'] in act_filter]
    
    if not filtered:
        return None, 0.0
    
    max_result = max(filtered, key=lambda x: x['gflops'])
    return max_result, max_result['gflops']


def create_plots(onednn_results, custom_gemm_results, output_prefix='benchmark', plot_title=None, suffix='', roofline_results=None):
    """Create bar plots comparing oneDNN and custom GEMM performance."""
    
    # Get all (M, N, K) combinations
    all_configs = sorted(set(list(onednn_results.keys()) + list(custom_gemm_results.keys())))
    
    # Prepare data for plotting
    data = []
    
    for M, N, K in all_configs:
        config_label = f"{M}x{N}x{K}"
        
        onednn_perf = onednn_results.get((M, N, K), 0.0)
        
        custom_results = custom_gemm_results.get((M, N, K), [])
        
        # Max for ACT = 0
        max_act_0, perf_act_0 = find_max_performance(custom_results, [0])
        
        data.append({
            'config': (M, N, K),
            'label': config_label,
            'onednn': onednn_perf,
            'custom_act0': perf_act_0,
            'custom_act0_cfg': max_act_0
        })
    
    # Split data into 4 groups
    group_size = (len(data) + 3) // 4
    data_groups = [
        data[i*group_size:(i+1)*group_size] 
        for i in range(4)
    ]
    
    # Create bar plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        width = 0.35
        
        # Convert GFLOPS to TFLOPS
        onednn_vals = [d['onednn'] / 1000.0 for d in plot_data]
        custom_act0_vals = [d['custom_act0'] / 1000.0 for d in plot_data]
        
        bars2 = ax.bar(x + width/2, custom_act0_vals, width, label='SFC-CA GEMM', alpha=0.8)
        bars1 = ax.bar(x - width/2, onednn_vals, width, label='oneDNN', alpha=0.8)
        
        # Add roofline as dashed line if available
        if roofline_results:
            roofline_vals = [roofline_results.get(d['config'], 0.0) / 1000.0 for d in plot_data]
            ax.plot(x, roofline_vals, linestyle='--', linewidth=1.5, label='Roofline', alpha=0.8, color='red')
        
        # Add value labels on top of bars
        for bar, val in zip(bars1, onednn_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        for bar, val in zip(bars2, custom_act0_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks every 10 TFLOPS
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_bar_plot{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot (4 parts): {output_file}")
    
    plt.close()
    
    # Create line plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        
        # Convert GFLOPS to TFLOPS
        onednn_vals = [d['onednn'] / 1000.0 for d in plot_data]
        custom_act0_vals = [d['custom_act0'] / 1000.0 for d in plot_data]
        
        # Plot lines with markers
        ax.plot(x, custom_act0_vals, marker='s', linewidth=1.5, markersize=4, 
                label='SFC-CA GEMM', alpha=0.8)
        ax.plot(x, onednn_vals, marker='o', linewidth=1.5, markersize=4, 
                label='oneDNN', alpha=0.8)
        
        # Add roofline as dashed line if available
        if roofline_results:
            roofline_vals = [roofline_results.get(d['config'], 0.0) / 1000.0 for d in plot_data]
            ax.plot(x, roofline_vals, linestyle='--', linewidth=1.5, label='Roofline', alpha=0.8, color='red')
        
        # Add value labels on top of line plot points
        for i, val in enumerate(onednn_vals):
            if val > 0:
                ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        for i, val in enumerate(custom_act0_vals):
            if val > 0:
                ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='both', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks every 10 TFLOPS
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_line_plot{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved line plot (4 parts): {output_file}")
    
    plt.close()
    
    # Create summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    if roofline_results:
        print(f"{'M':<6} {'N':<6} {'K':<6} {'oneDNN':<10} {'Custom(0)':<12} {'Roofline':<12} {'% of Roof':<10} {'Config(0)':<20}")
    else:
        print(f"{'M':<6} {'N':<6} {'K':<6} {'oneDNN':<10} {'Custom(0)':<12} {'Config(0)':<20}")
    print("-"*80)
    
    roofline_efficiencies = []
    for d in data:
        M, N, K = d['config']
        cfg0 = d['custom_act0_cfg']
        
        cfg0_str = f"BFK={cfg0['BFK']},C={cfg0['C']},ACT={cfg0['ACT']}" if cfg0 else "N/A"
        
        if roofline_results:
            roofline_perf = roofline_results.get((M, N, K), 0.0)
            if roofline_perf > 0 and d['custom_act0'] > 0:
                efficiency = (d['custom_act0'] / roofline_perf) * 100.0
                roofline_efficiencies.append(efficiency)
                print(f"{M:<6} {N:<6} {K:<6} {d['onednn']:<10.1f} {d['custom_act0']:<12.1f} {roofline_perf:<12.1f} {efficiency:<10.1f} {cfg0_str:<20}")
            else:
                print(f"{M:<6} {N:<6} {K:<6} {d['onednn']:<10.1f} {d['custom_act0']:<12.1f} {roofline_perf:<12.1f} {'N/A':<10} {cfg0_str:<20}")
        else:
            print(f"{M:<6} {N:<6} {K:<6} {d['onednn']:<10.1f} {d['custom_act0']:<12.1f} {cfg0_str:<20}")
    
    print("="*80)
    
    # Print roofline efficiency summary
    if roofline_results and roofline_efficiencies:
        avg_efficiency = np.mean(roofline_efficiencies)
        min_efficiency = np.min(roofline_efficiencies)
        max_efficiency = np.max(roofline_efficiencies)
        print("\nROOFLINE EFFICIENCY SUMMARY")
        print("="*80)
        print(f"Average SFC-CA GEMM efficiency (% of Roofline): {avg_efficiency:.1f}%")
        print(f"Minimum efficiency: {min_efficiency:.1f}%")
        print(f"Maximum efficiency: {max_efficiency:.1f}%")
        print(f"Average gap to roofline: {100.0 - avg_efficiency:.1f}%")
        print("="*80)


def create_plots_by_compute_intensity(onednn_results, custom_gemm_results, output_prefix='benchmark', plot_title=None, suffix='', roofline_results=None):
    """Create plots sorted by computational intensity."""
    
    # Get all (M, N, K) combinations
    all_configs = sorted(set(list(onednn_results.keys()) + list(custom_gemm_results.keys())))
    
    # Prepare data for plotting
    data = []
    
    for M, N, K in all_configs:
        config_label = f"{M}x{N}x{K}"
        
        # Calculate computational intensity: (2*M*N*K)/(2*M*N+2*M*K+2*K*N)
        comp_intensity = (2.0 * M * N * K) / (2.0 * M * N + 2.0 * M * K + 2.0 * K * N)
        
        onednn_perf = onednn_results.get((M, N, K), 0.0)
        
        custom_results = custom_gemm_results.get((M, N, K), [])
        
        # Max for ACT = 0
        max_act_0, perf_act_0 = find_max_performance(custom_results, [0])
        
        data.append({
            'config': (M, N, K),
            'label': config_label,
            'comp_intensity': comp_intensity,
            'onednn': onednn_perf,
            'custom_act0': perf_act_0,
            'custom_act0_cfg': max_act_0
        })
    
    # Sort by computational intensity
    data_sorted = sorted(data, key=lambda x: x['comp_intensity'])
    
    # Split data into 4 groups
    group_size = (len(data_sorted) + 3) // 4
    data_groups = [
        data_sorted[i*group_size:(i+1)*group_size] 
        for i in range(4)
    ]
    
    # Create bar plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        width = 0.35
        
        # Convert GFLOPS to TFLOPS
        onednn_vals = [d['onednn'] / 1000.0 for d in plot_data]
        custom_act0_vals = [d['custom_act0'] / 1000.0 for d in plot_data]
        
        bars2 = ax.bar(x + width/2, custom_act0_vals, width, label='SFC-CA GEMM', alpha=0.8)
        bars1 = ax.bar(x - width/2, onednn_vals, width, label='oneDNN', alpha=0.8)
        
        # Add roofline as dashed line if available
        if roofline_results:
            roofline_vals = [roofline_results.get(d['config'], 0.0) / 1000.0 for d in plot_data]
            ax.plot(x, roofline_vals, linestyle='--', linewidth=1.5, label='Roofline', alpha=0.8, color='red')
        
        # Add value labels on top of bars
        for bar, val in zip(bars1, onednn_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        for bar, val in zip(bars2, custom_act0_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks every 10 TFLOPS
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_by_comp_intensity_bar{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved computational intensity bar plot (4 parts): {output_file}")
    
    plt.close()
    
    # Create line plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        
        # Convert GFLOPS to TFLOPS
        onednn_vals = [d['onednn'] / 1000.0 for d in plot_data]
        custom_act0_vals = [d['custom_act0'] / 1000.0 for d in plot_data]
        
        # Plot lines with markers
        ax.plot(x, custom_act0_vals, marker='s', linewidth=1.5, markersize=4, 
                label='SFC-CA GEMM', alpha=0.8)
        ax.plot(x, onednn_vals, marker='o', linewidth=1.5, markersize=4, 
                label='oneDNN', alpha=0.8)
        
        # Add roofline as dashed line if available
        if roofline_results:
            roofline_vals = [roofline_results.get(d['config'], 0.0) / 1000.0 for d in plot_data]
            ax.plot(x, roofline_vals, linestyle='--', linewidth=1.5, label='Roofline', alpha=0.8, color='red')
        
        # Add value labels on top of line plot points
        for i, val in enumerate(onednn_vals):
            if val > 0:
                ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        for i, val in enumerate(custom_act0_vals):
            if val > 0:
                ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='both', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks every 10 TFLOPS
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_by_comp_intensity_line{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved computational intensity line plot (4 parts): {output_file}")
    
    plt.close()


def create_speedup_plot_by_compute_intensity(onednn_results, custom_gemm_results, output_prefix='benchmark', plot_title=None, suffix=''):
    """Create speedup plot sorted by computational intensity."""
    
    # Get all (M, N, K) combinations
    all_configs = sorted(set(list(onednn_results.keys()) + list(custom_gemm_results.keys())))
    
    # Prepare data for speedup calculation
    data = []
    speedups_act0 = []
    
    for M, N, K in all_configs:
        config_label = f"{M}x{N}x{K}"
        
        # Calculate computational intensity
        comp_intensity = (2.0 * M * N * K) / (2.0 * M * N + 2.0 * M * K + 2.0 * K * N)
        
        onednn_perf = onednn_results.get((M, N, K), 0.0)
        
        if onednn_perf == 0.0:
            continue  # Skip if no oneDNN baseline
        
        custom_results = custom_gemm_results.get((M, N, K), [])
        
        # Max for ACT = 0
        max_act_0, perf_act_0 = find_max_performance(custom_results, [0])
        
        # Calculate speedup (speedup = Custom_GFLOPS / oneDNN_GFLOPS)
        speedup_act0 = perf_act_0 / onednn_perf if perf_act_0 > 0 else 0.0
        
        data.append({
            'config': (M, N, K),
            'label': config_label,
            'comp_intensity': comp_intensity,
            'speedup_act0': speedup_act0
        })
        
        if speedup_act0 > 0:
            speedups_act0.append(speedup_act0)
    
    if not data:
        print("No valid speedup data to plot")
        return
    
    # Sort by computational intensity
    data_sorted = sorted(data, key=lambda x: x['comp_intensity'])
    
    # Calculate geometric and arithmetic means
    geomean_act0 = np.exp(np.mean(np.log(speedups_act0))) if speedups_act0 else 0.0
    mean_act0 = np.mean(speedups_act0) if speedups_act0 else 0.0
    
    # Split data into 4 groups
    group_size = (len(data_sorted) + 3) // 4
    data_groups = [
        data_sorted[i*group_size:(i+1)*group_size] 
        for i in range(4)
    ]
    
    # Create speedup plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        
        # Plot baseline (oneDNN) at y=1
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='oneDNN (baseline)', alpha=0.7)
        
        # Plot speedup line
        speedup_act0_vals = [d['speedup_act0'] for d in plot_data]
        
        ax.plot(x, speedup_act0_vals, marker='o', linewidth=1.5, markersize=5, 
                label=f'SFC-CA GEMM - GeoMean: {geomean_act0:.3f}x', alpha=0.8)
        
        # Add value labels on points
        for i, v0 in enumerate(speedup_act0_vals):
            if v0 > 0:
                ax.text(i, v0, f'{v0:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Speedup over oneDNN', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='both', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks for speedup (smaller intervals)
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_speedup_by_comp_intensity{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved computational intensity speedup plot (4 parts): {output_file}")
    
    plt.close()


def create_speedup_plot(onednn_results, custom_gemm_results, output_prefix='benchmark', plot_title=None, suffix=''):
    """Create speedup plot comparing Custom GEMM against oneDNN baseline."""
    
    # Get all (M, N, K) combinations
    all_configs = sorted(set(list(onednn_results.keys()) + list(custom_gemm_results.keys())))
    
    # Prepare data for speedup calculation
    data = []
    speedups_act0 = []
    
    for M, N, K in all_configs:
        config_label = f"{M}x{N}x{K}"
        
        onednn_perf = onednn_results.get((M, N, K), 0.0)
        
        if onednn_perf == 0.0:
            continue  # Skip if no oneDNN baseline
        
        custom_results = custom_gemm_results.get((M, N, K), [])
        
        # Max for ACT = 0
        max_act_0, perf_act_0 = find_max_performance(custom_results, [0])
        
        # Calculate speedup (speedup = Custom_GFLOPS / oneDNN_GFLOPS)
        speedup_act0 = perf_act_0 / onednn_perf if perf_act_0 > 0 else 0.0
        
        data.append({
            'config': (M, N, K),
            'label': config_label,
            'speedup_act0': speedup_act0
        })
        
        if speedup_act0 > 0:
            speedups_act0.append(speedup_act0)
    
    if not data:
        print("No valid speedup data to plot")
        return
    
    # Calculate geometric and arithmetic means
    geomean_act0 = np.exp(np.mean(np.log(speedups_act0))) if speedups_act0 else 0.0
    mean_act0 = np.mean(speedups_act0) if speedups_act0 else 0.0
    
    # Split data into 4 groups
    group_size = (len(data) + 3) // 4
    data_groups = [
        data[i*group_size:(i+1)*group_size] 
        for i in range(4)
    ]
    
    # Create speedup plots - stack all 4 parts in one PDF
    fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT * 4))
    
    for group_idx, (ax, plot_data) in enumerate(zip(axes, data_groups)):
        if not plot_data:
            ax.axis('off')
            continue
            
        x = np.arange(len(plot_data))
        
        # Plot baseline (oneDNN) at y=1
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='oneDNN (baseline)', alpha=0.7)
        
        # Plot speedup line
        speedup_act0_vals = [d['speedup_act0'] for d in plot_data]
        
        ax.plot(x, speedup_act0_vals, marker='o', linewidth=1.5, markersize=5, 
                label=f'SFC-CA GEMM - GeoMean: {geomean_act0:.3f}x', alpha=0.8)
        
        # Add value labels on points
        for i, v0 in enumerate(speedup_act0_vals):
            if v0 > 0:
                ax.text(i, v0, f'{v0:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        # Customize plot
        if plot_title and group_idx == 0:
            ax.set_title(plot_title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Speedup over oneDNN', fontsize=6, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['label'] for d in plot_data], rotation=25, ha='right', fontsize=4)
        if group_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(axis='both', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Set y-axis ticks for speedup (smaller intervals)
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis='y', labelsize=5)
        
    plt.tight_layout()
    
    # Save all 4 parts as single PDF
    output_file = f'{output_prefix}_speedup_plot{suffix}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved speedup plot (4 parts): {output_file}")
    
    plt.close()
    
    # Print speedup summary
    print("\n" + "="*80)
    print("SPEEDUP SUMMARY (vs oneDNN)")
    print("="*80)
    print(f"{'M':<6} {'N':<6} {'K':<6} {'Custom(0) Speedup':<20}")
    print("-"*80)
    
    for d in data:
        M, N, K = d['config']
        print(f"{M:<6} {N:<6} {K:<6} {d['speedup_act0']:<20.3f}")
    
    print("-"*80)
    print(f"{'Arithmetic Mean:':<18} {mean_act0:>20.3f}x")
    print(f"{'Geometric Mean:':<18} {geomean_act0:>20.3f}x")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark_paper.py <benchmark_results_file> [output_prefix] [plot_title] [suffix] [roofline_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'benchmark_paper'
    plot_title = sys.argv[3] if len(sys.argv) > 3 else None
    suffix = sys.argv[4] if len(sys.argv) > 4 else ''
    roofline_file = sys.argv[5] if len(sys.argv) > 5 else None
    
    print(f"Parsing benchmark results from: {input_file}")
    onednn_results, custom_gemm_results = parse_benchmark_results(input_file)
    
    print(f"Found {len(onednn_results)} oneDNN results")
    print(f"Found {sum(len(v) for v in custom_gemm_results.values())} custom GEMM results")
    
    # Parse roofline results if provided
    roofline_results = None
    if roofline_file:
        print(f"\nParsing roofline results from: {roofline_file}")
        roofline_results = parse_roofline_results(roofline_file)
        print(f"Found {len(roofline_results)} roofline results")
    
    print("\nCreating plots (21cm width x 12cm height, 4 parts stacked in single PDF)...")
    create_plots(onednn_results, custom_gemm_results, output_prefix, plot_title, suffix, roofline_results)
    
    print("\nCreating speedup analysis...")
    create_speedup_plot(onednn_results, custom_gemm_results, output_prefix, plot_title, suffix)
    
    print("\nCreating plots sorted by computational intensity...")
    create_plots_by_compute_intensity(onednn_results, custom_gemm_results, output_prefix, plot_title, suffix, roofline_results)
    create_speedup_plot_by_compute_intensity(onednn_results, custom_gemm_results, output_prefix, plot_title, suffix)
    
    print("\nAnalysis complete!")
    print(f"All plots have dimensions: width = 21cm (8.27 inches), height = 4cm (1.57 inches)")
    print(f"Each plot type has 4 parts stacked vertically in a single PDF file.")
    print(f"Performance values are shown in TFLOPS (converted from GFLOPS).")
    print(f"Generated 6 PDF files total (2 regular + 2 comp. intensity + 2 speedup plots).")


if __name__ == "__main__":
    main()
