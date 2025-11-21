#!/usr/bin/env python3
"""
Parse a benchmarking output file and plot effective GFLOPS vs sparsity.
Usage: python3 plot_gflops.py benchmark.out -o plot.png --dense_perf X --title="My plot" --metric gflops/bw
"""
import re
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_output(path):
    text = Path(path).read_text()
    # Each block starts with the command line containing the sparsity as the 8th numeric arg
    # Example command line contains: ... 0.0 1 ... (sparsity)
    # We'll extract all occurrences of 'GFLOPS for sparse' and associate the nearest preceding sparsity value

    # Find all command lines that start with ./spgemm or contain './spgemm'
    cmd_re = re.compile(r"^.*\./spgemm.*$", re.MULTILINE)
    gflops_re = re.compile(r"([0-9]+\.?[0-9]*)\s+GFLOPS for sparse")
    gbps_re = re.compile(r"Effective\s+GB/s\s+is\s+([0-9]+\.?[0-9]*)")

    # Extract command lines and their positions
    cmds = []
    for m in cmd_re.finditer(text):
        cmds.append((m.start(), m.group(0)))

    # For each command, look for GB/s and GFLOPS in the text segment until the next command
    results = []
    for i, (cpos, cstr) in enumerate(cmds):
        start = cpos
        end = cmds[i+1][0] if i+1 < len(cmds) else len(text)
        seg = text[start:end]

        # Extract numeric args from command to find sparsity
        nums = re.findall(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", cstr)
        sparsity = None
        for tok in nums:
            try:
                v = float(tok)
            except:
                continue
            if 0.0 <= v <= 1.0:
                sparsity = v
                break
        # Also try to find a '(XX sparsity)' inside the segment
        nz_spars_m = re.search(r"with\s+\d+\s+NZ entries \((\d+)[^)]*sparsity\)", seg)
        if nz_spars_m:
            try:
                sparsity = float(nz_spars_m.group(1)) / 100.0
            except:
                pass

        # Find GFLOPS and GB/s in this segment
        gflops_m = gflops_re.search(seg)
        gbps_m = gbps_re.search(seg)
        gval = float(gflops_m.group(1)) if gflops_m else None
        bwval = float(gbps_m.group(1)) if gbps_m else None
        results.append((sparsity, gval, bwval))

    # Remove entries without sparsity and sort
    final = [(s, g, b) for (s, g, b) in results if s is not None]
    final.sort(key=lambda x: x[0])
    return final


def plot(results, out=None):
    if not results:
        print("No data parsed")
        return
    # results are (sparsity, gflops, gbps)
    # Filter out extremely high sparsity (e.g. 0.99) if present
    filtered = [(s, g, b) for s, g, b in results if s < 0.99]
    if not filtered:
        print("No data left after filtering sparsity >= 0.99")
        return
    metric = getattr(plt, '_metric', 'gflops')
    sparsities = [s*100.0 for s, _, _ in filtered]
    if metric == 'gflops':
        values = [g for _, g, _ in filtered]
    else:
        values = [b for _, _, b in filtered]
    fig, ax = plt.subplots(figsize=(8,4))
    labels = [f"{int(s)}%" for s in sparsities]
    bars = ax.bar(labels, values)
    # Add numeric labels on top of bars
    for rect, val in zip(bars, values):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"{val:.1f}",
                 ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Sparsity')
    # Set left axis label according to metric
    metric = getattr(plt, '_metric', 'gflops')
    if metric == 'bw':
        ax.set_ylabel('Effective GB/s')
    else:
        ax.set_ylabel('Effective GFLOPS')
    # Use custom title if provided
    plot_title = getattr(plt, '_plot_title', None)
    if plot_title is not None:
        ax.set_title(plot_title)
    else:
        ax.set_title('GFLOPS vs sparsity')
    # Draw dense performance horizontal line if requested (passed via main)
    dense_perf = getattr(plt, '_dense_perf', None)
    if dense_perf is not None:
        ax.axhline(y=dense_perf, color='red', linestyle='--', linewidth=1)
        # place label near the right end of the line
        xlim = ax.get_xlim()
        # xpos in data coordinates: use right-most bar index
        xpos = len(labels) - 0.5
        ax.text(xpos, dense_perf, ' dense GEMM', color='red', va='center', ha='left')

        # Secondary y-axis for speedup = measured_metric / dense_perf
        ax2 = ax.twinx()
        ax2.set_ylabel('Speedup over dense')
        # compute speedups safely (skip if value is None or dense_perf is None)
        speedups = []
        for v in values:
            if dense_perf is None or v is None:
                speedups.append(None)
            else:
                try:
                    speedups.append(float(v) / float(dense_perf))
                except Exception:
                    speedups.append(None)
        # plot markers on secondary axis aligned with bar centers
        x_positions = list(range(len(labels)))
        # filter out None speedups (if missing data)
        x_plot = [x for x, sp in zip(x_positions, speedups) if sp is not None]
        sp_plot = [sp for sp in speedups if sp is not None]
        if sp_plot:
            ax2.plot(x_plot, sp_plot, color='blue', marker='o', linestyle='-', linewidth=1)
            # add labels for each point below the marker
            for x, sp in zip(x_plot, sp_plot):
                ax2.annotate(f"{sp:.2f}", xy=(x, sp), xytext=(0, -10), textcoords='offset points',
                             color='blue', ha='center', va='top', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if out:
        fig.savefig(out)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot GFLOPS vs sparsity from a benchmark output file')
    parser.add_argument('file', help='Benchmark output file')
    parser.add_argument('--out', '-o', help='Save plot to file (png)')
    parser.add_argument('--dense_perf', type=float, default=None, help='Draw horizontal line at this GFLOPS value labeled "dense GEMM"')
    parser.add_argument('--title', dest='title', default=None, help='Add this title to the plot')
    parser.add_argument('--metric', choices=['gflops', 'bw'], default='gflops', help='Metric to plot: gflops or bw (GB/s)')
    args = parser.parse_args()
    results = parse_output(args.file)
    # pass dense_perf to plot via plt attribute (simple approach)
    if args.dense_perf is not None:
        setattr(plt, '_dense_perf', args.dense_perf)
    setattr(plt, '_metric', args.metric)
    # pass title via plt attribute as well
    if args.title is not None:
        setattr(plt, '_plot_title', args.title)
    plot(results, args.out)

if __name__ == '__main__':
    main()
