#!/usr/bin/env python
"""Figure C: AR Representation Comparison.

Reads metrics_generation.csv and creates visualization comparing
generation quality across AR representations (SMILES, SMILES_BPE,
SELFIES, Group_SELFIES).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ar_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create AR representation comparison plots."""
    if df.empty:
        print("No generation data found. Skipping plot.")
        return

    # Filter to AR_Transformer method; fall back to all data if method column absent
    ar_df = df[df['method'] == 'AR_Transformer'].copy() if 'method' in df.columns else df.copy()
    if ar_df.empty:
        ar_df = df.copy()

    # Aggregate by representation (mean over model sizes)
    grouped = ar_df.groupby('representation').agg({
        'validity': 'mean',
        'validity_two_stars': 'mean',
        'novelty': 'mean',
        'uniqueness': 'mean',
        'avg_diversity': 'mean',
        'valid_per_sec': 'mean',
        'mean_sa': 'mean',
    }).reset_index()

    reps = list(grouped['representation'])
    x = np.arange(len(reps))
    rep_colors = {
        'SMILES': '#1f77b4',
        'SMILES_BPE': '#aec7e8',
        'SELFIES': '#ff7f0e',
        'Group_SELFIES': '#2ca02c',
    }
    colors = [rep_colors.get(r, 'gray') for r in reps]
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Validity (2-star)
    ax = axes[0, 0]
    ax.bar(x, grouped['validity_two_stars'], color=colors)
    ax.set_xlabel('Representation')
    ax.set_ylabel('Validity (2-star)')
    ax.set_title('Validity Across AR Representations')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Throughput
    ax = axes[0, 1]
    ax.bar(x, grouped['valid_per_sec'], color=colors)
    ax.set_xlabel('Representation')
    ax.set_ylabel('Valid Samples / Second')
    ax.set_title('Throughput Across AR Representations')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Novelty and Uniqueness
    ax = axes[1, 0]
    ax.bar(x - width / 2, grouped['novelty'], width, label='Novelty', color='#1f77b4')
    ax.bar(x + width / 2, grouped['uniqueness'], width, label='Uniqueness', color='#ff7f0e')
    ax.set_xlabel('Representation')
    ax.set_ylabel('Fraction')
    ax.set_title('Novelty & Uniqueness Across AR Representations')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Diversity
    ax = axes[1, 1]
    ax.bar(x, grouped['avg_diversity'], color=colors)
    ax.set_xlabel('Representation')
    ax.set_ylabel('Average Diversity')
    ax.set_title('Diversity Across AR Representations')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure C to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure C: AR Representation Comparison")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_generation.csv",
                        help="Path to metrics_generation.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_c_ar_comparison.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_ar_comparison(df, output_path)


if __name__ == "__main__":
    main()
