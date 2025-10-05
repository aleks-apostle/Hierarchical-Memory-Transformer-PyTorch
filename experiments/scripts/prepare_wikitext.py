#!/usr/bin/env python3
"""
WikiText-103 Data Preparation Script

This script:
1. Downloads WikiText-103 from HuggingFace
2. Generates dataset statistics
3. Caches preprocessed data for training
4. Creates visualizations and reports

Usage:
    python experiments/scripts/prepare_wikitext.py [--cache-dir DATA_DIR] [--max-length MAX_LEN]

Example:
    python experiments/scripts/prepare_wikitext.py --cache-dir ./data/wikitext103
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from hmt.data import WikiTextDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare WikiText-103 dataset for HMT training")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/wikitext103",
        help="Directory to cache preprocessed data (default: ./data/wikitext103)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum article length in tokens (default: None, no limit)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=128,
        help="Minimum article length in tokens (default: 128)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    return parser.parse_args()


def analyze_dataset(dataset: WikiTextDataset, split_name: str) -> dict:
    """
    Analyze dataset and compute statistics.

    Args:
        dataset: WikiTextDataset instance
        split_name: Name of the split ('train', 'validation', 'test')

    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {split_name} split...")
    print(f"{'='*80}")

    lengths = []
    for article in tqdm(dataset.articles, desc="Computing statistics"):
        lengths.append(len(article['input_ids']))

    lengths = np.array(lengths)

    stats = {
        'split': split_name,
        'num_articles': len(lengths),
        'total_tokens': int(lengths.sum()),
        'mean_length': float(lengths.mean()),
        'median_length': float(np.median(lengths)),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'std_length': float(lengths.std()),
        'percentiles': {
            '25': float(np.percentile(lengths, 25)),
            '50': float(np.percentile(lengths, 50)),
            '75': float(np.percentile(lengths, 75)),
            '90': float(np.percentile(lengths, 90)),
            '95': float(np.percentile(lengths, 95)),
            '99': float(np.percentile(lengths, 99)),
        },
        'length_distribution': lengths.tolist(),  # For visualization
    }

    # Print statistics
    print(f"\n📊 Statistics for {split_name}:")
    print(f"  Articles:     {stats['num_articles']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Mean length:  {stats['mean_length']:.1f} tokens")
    print(f"  Median:       {stats['median_length']:.1f} tokens")
    print(f"  Min/Max:      {stats['min_length']} / {stats['max_length']:,} tokens")
    print(f"  Std dev:      {stats['std_length']:.1f}")
    print(f"\n  Percentiles:")
    for p, val in stats['percentiles'].items():
        print(f"    {p:>3}th: {val:>7.1f} tokens")

    # HMT-specific analysis
    segment_length = 512
    gpt2_max_length = 1024

    articles_exceeding_gpt2 = (lengths > gpt2_max_length).sum()
    pct_exceeding = (articles_exceeding_gpt2 / len(lengths)) * 100

    avg_segments = (lengths / segment_length).mean()

    print(f"\n  🎯 HMT Relevance:")
    print(f"    Articles > 1024 tokens (GPT-2 limit): {articles_exceeding_gpt2:,} ({pct_exceeding:.1f}%)")
    print(f"    Average segments per article (L={segment_length}): {avg_segments:.1f}")

    return stats


def visualize_statistics(all_stats: list, output_dir: Path):
    """
    Create visualization plots for dataset statistics.

    Args:
        all_stats: List of statistics dictionaries from analyze_dataset
        output_dir: Directory to save plots
    """
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Length distribution comparison across splits
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, stats in enumerate(all_stats):
        ax = axes[i]
        lengths = np.array(stats['length_distribution'])

        ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(1024, color='red', linestyle='--', linewidth=2, label='GPT-2 limit')
        ax.set_xlabel('Article Length (tokens)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{stats['split'].capitalize()} Split")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'length_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'length_distribution.png'}")
    plt.close()

    # 2. CDF plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for stats in all_stats:
        lengths = np.array(stats['length_distribution'])
        sorted_lengths = np.sort(lengths)
        cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        ax.plot(sorted_lengths, cdf, linewidth=2, label=stats['split'].capitalize())

    ax.axvline(1024, color='red', linestyle='--', linewidth=2, label='GPT-2 limit', alpha=0.7)
    ax.set_xlabel('Article Length (tokens)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Article Lengths')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'length_cdf.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'length_cdf.png'}")
    plt.close()

    # 3. Segment count distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    segment_length = 512
    for stats in all_stats:
        lengths = np.array(stats['length_distribution'])
        num_segments = np.ceil(lengths / segment_length)

        segment_counts = np.arange(1, num_segments.max() + 1)
        frequencies = [(num_segments >= count).sum() for count in segment_counts]

        ax.plot(segment_counts, frequencies, linewidth=2, marker='o',
                label=f"{stats['split'].capitalize()} ({stats['num_articles']} articles)")

    ax.set_xlabel(f'Number of Segments (L={segment_length})')
    ax.set_ylabel('Number of Articles')
    ax.set_title('Articles Requiring N Segments for HMT Processing')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'segment_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'segment_distribution.png'}")
    plt.close()

    # 4. Summary comparison table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ['Split', 'Articles', 'Total Tokens', 'Mean', 'Median', 'Max', '>1024 (%)']
    rows = []

    for stats in all_stats:
        lengths = np.array(stats['length_distribution'])
        pct_over_1024 = ((lengths > 1024).sum() / len(lengths)) * 100

        rows.append([
            stats['split'].capitalize(),
            f"{stats['num_articles']:,}",
            f"{stats['total_tokens']:,}",
            f"{stats['mean_length']:.0f}",
            f"{stats['median_length']:.0f}",
            f"{stats['max_length']:,}",
            f"{pct_over_1024:.1f}%"
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('WikiText-103 Dataset Summary', fontsize=14, weight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'summary_table.png'}")
    plt.close()


def main():
    args = parse_args()

    print("="*80)
    print("WikiText-103 Data Preparation for HMT")
    print("="*80)

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCache directory: {cache_dir.absolute()}")

    # Initialize tokenizer
    print("\nInitializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocabulary size: {len(tokenizer)}")

    # Load and analyze all splits
    all_stats = []
    for split in ['train', 'validation', 'test']:
        dataset = WikiTextDataset(
            split=split,
            tokenizer=tokenizer,
            min_length=args.min_length,
            max_length=args.max_length,
            cache_dir=str(cache_dir),
        )

        stats = analyze_dataset(dataset, split)
        all_stats.append(stats)

    # Save statistics
    stats_file = cache_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        # Remove length_distribution from saved stats (too large)
        stats_to_save = []
        for stats in all_stats:
            stats_copy = stats.copy()
            stats_copy.pop('length_distribution')
            stats_to_save.append(stats_copy)

        json.dump(stats_to_save, f, indent=2)

    print(f"\n✅ Saved statistics to: {stats_file}")

    # Generate visualizations
    if args.visualize:
        visualize_statistics(all_stats, cache_dir / 'visualizations')

    # Summary
    print(f"\n{'='*80}")
    print("✅ Data preparation complete!")
    print(f"{'='*80}")
    print(f"\nDataset ready for HMT training:")
    print(f"  Cache: {cache_dir.absolute()}")
    print(f"  Statistics: {stats_file}")
    if args.visualize:
        print(f"  Visualizations: {cache_dir / 'visualizations'}")

    print(f"\n🚀 Next steps:")
    print(f"  1. Review the statistics and visualizations")
    print(f"  2. Run tests: pytest tests/test_data.py")
    print(f"  3. Start implementing HMT memory components!")


if __name__ == "__main__":
    main()
