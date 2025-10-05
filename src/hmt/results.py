"""
HMT Results Management - Local-Only Experiment Tracking

This module provides local experiment tracking and result management without
any cloud dependencies (no W&B, MLflow, etc.). All results are stored locally
in structured directories with JSON/CSV files and matplotlib plots.

Paper Reference: Section 4 (Experiments), Tables 1-7, Figures 4-5
Used for reproducing all paper results and storing evaluation metrics locally.
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment run.

    Tracks all hyperparameters and settings for reproducibility.
    """
    name: str
    model_name: str
    dataset: str
    segment_length: int
    num_memory_embeddings: int
    sensory_memory_size: int
    batch_size: int
    learning_rate: float
    seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    paper_table: Optional[str] = None  # e.g., "Table 1", "Figure 4"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: Path):
        """Save configuration to YAML or JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ResultLogger:
    """
    Local-only experiment result tracker.

    Manages experiment directories, logs metrics to JSON/CSV, generates plots,
    and creates markdown reports. All operations are local - no cloud services.

    Directory Structure:
        results/
        ├── experiments/
        │   ├── exp_001_table1_opt350m/
        │   │   ├── config.json
        │   │   ├── metrics.json
        │   │   ├── metrics.csv
        │   │   ├── plots/
        │   │   │   ├── loss_curve.png
        │   │   │   ├── perplexity.png
        │   │   │   └── memory_stats.png
        │   │   ├── checkpoints/
        │   │   │   └── best_model.pt
        │   │   └── report.md
        │   └── exp_002_figure4_long_context/
        └── paper_reproduction/
            ├── table1.csv
            ├── table6_ablations.csv
            ├── figure4_data.csv
            └── plots/

    Example:
        >>> logger = ResultLogger(experiment_name="table1_reproduction")
        >>> logger.log_metrics(step=100, loss=2.5, perplexity=12.3)
        >>> logger.plot_metrics(["loss", "perplexity"])
        >>> logger.save_checkpoint(model.state_dict(), is_best=True)
        >>> logger.generate_report()
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: Union[str, Path] = "results",
        config: Optional[ExperimentConfig] = None,
        auto_increment: bool = True,
    ):
        """
        Initialize result logger.

        Args:
            experiment_name: Name for this experiment
            base_dir: Base directory for all results (default: "results/")
            config: Experiment configuration to save
            auto_increment: Auto-increment experiment number if name exists
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.config = config

        # Create experiment directory with auto-increment
        self.exp_dir = self._create_experiment_dir(experiment_name, auto_increment)

        # Create subdirectories
        self.plots_dir = self.exp_dir / "plots"
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.plots_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Initialize metric storage
        self.metrics: List[Dict[str, Any]] = []
        self.metrics_file = self.exp_dir / "metrics.json"
        self.metrics_csv = self.exp_dir / "metrics.csv"

        # Save configuration
        if config is not None:
            config.save(self.exp_dir / "config.json")

        log.info(f"ResultLogger initialized: {self.exp_dir}")

    def _create_experiment_dir(self, name: str, auto_increment: bool) -> Path:
        """Create experiment directory with optional auto-increment."""
        experiments_dir = self.base_dir / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)

        if auto_increment:
            # Find next available number
            existing = list(experiments_dir.glob(f"exp_*_{name}"))
            if existing:
                # Extract numbers and find max
                numbers = []
                for p in existing:
                    parts = p.name.split('_')
                    if parts[0] == 'exp' and parts[1].isdigit():
                        numbers.append(int(parts[1]))
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1

            exp_name = f"exp_{next_num:03d}_{name}"
        else:
            exp_name = name

        exp_dir = experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        return exp_dir

    def log_metrics(self, step: int, **metrics: float):
        """
        Log metrics for a training/evaluation step.

        Args:
            step: Current step/iteration number
            **metrics: Metric name-value pairs (e.g., loss=2.5, ppl=12.3)

        Example:
            >>> logger.log_metrics(step=100, loss=2.5, perplexity=12.3, lr=1e-5)
        """
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.metrics.append(entry)

        # Save to JSON (append mode)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save to CSV
        self._save_metrics_csv()

        # Log to console
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        log.info(f"Step {step}: {metrics_str}")

    def _save_metrics_csv(self):
        """Save metrics to CSV file."""
        if not self.metrics:
            return

        # Get all unique keys
        all_keys = set()
        for m in self.metrics:
            all_keys.update(m.keys())

        fieldnames = sorted(all_keys)

        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)

    def plot_metrics(
        self,
        metric_names: List[str],
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        show: bool = False,
    ):
        """
        Plot metrics over training steps.

        Args:
            metric_names: List of metric names to plot
            title: Plot title (auto-generated if None)
            save_name: Filename to save plot (auto-generated if None)
            show: Whether to display plot interactively

        Example:
            >>> logger.plot_metrics(["loss", "perplexity"])
            >>> logger.plot_metrics(["lr"], title="Learning Rate Schedule")
        """
        if not self.metrics:
            log.warning("No metrics to plot")
            return

        # Extract data
        df = pd.DataFrame(self.metrics)

        # Create figure
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, metric_names):
            if metric_name not in df.columns:
                log.warning(f"Metric '{metric_name}' not found in logged metrics")
                continue

            ax.plot(df['step'], df[metric_name], linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} vs Step')
            ax.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"metrics_{'_'.join(metric_names)}.png"

        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_comparison(
        self,
        data: Dict[str, List[float]],
        x_values: Optional[List[float]] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Comparison",
        save_name: str = "comparison.png",
        show: bool = False,
    ):
        """
        Plot comparison of multiple series (e.g., HMT vs baselines).

        Args:
            data: Dictionary mapping series names to value lists
            x_values: X-axis values (uses indices if None)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Example:
            >>> data = {
            ...     "HMT": [12.5, 11.8, 11.2],
            ...     "Vanilla": [14.2, 15.1, 16.3],
            ...     "Sliding Window": [13.1, 13.8, 14.5],
            ... }
            >>> logger.plot_comparison(
            ...     data, x_values=[1024, 2048, 4096],
            ...     xlabel="Sequence Length", ylabel="Perplexity",
            ...     title="HMT vs Baselines (Figure 4)", save_name="figure4.png"
            ... )
        """
        plt.figure(figsize=(10, 6))

        if x_values is None:
            x_values = list(range(len(next(iter(data.values())))))

        for name, values in data.items():
            plt.plot(x_values, values, marker='o', linewidth=2, label=name, markersize=6)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Comparison plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_table(
        self,
        data: Union[Dict, pd.DataFrame],
        filename: str,
        format: str = "csv",
    ):
        """
        Save tabular data to CSV or JSON.

        Args:
            data: Dictionary or DataFrame to save
            filename: Output filename
            format: "csv" or "json"

        Example:
            >>> results = {
            ...     "Model": ["HMT", "Vanilla", "RMT"],
            ...     "PPL": [21.3, 24.1, 22.8],
            ...     "Speed": [1.5, 1.0, 1.3],
            ... }
            >>> logger.save_table(results, "table1_results.csv")
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        filepath = self.exp_dir / filename

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        log.info(f"Table saved: {filepath}")

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        is_best: bool = False,
        additional_info: Optional[Dict] = None,
    ):
        """
        Save model checkpoint locally.

        Args:
            state_dict: Model state dictionary
            step: Current training step
            is_best: Whether this is the best model so far
            additional_info: Additional metadata to save

        Example:
            >>> logger.save_checkpoint(
            ...     model.state_dict(),
            ...     step=1000,
            ...     is_best=True,
            ...     additional_info={"perplexity": 21.3, "loss": 3.06}
            ... )
        """
        import torch

        checkpoint = {
            "step": step,
            "state_dict": state_dict,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_info:
            checkpoint.update(additional_info)

        # Save regular checkpoint
        ckpt_path = self.checkpoints_dir / f"checkpoint_step{step}.pt"
        torch.save(checkpoint, ckpt_path)
        log.info(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            log.info(f"Best model saved: {best_path}")

    def generate_report(self, save_name: str = "report.md"):
        """
        Generate a markdown report summarizing experiment.

        Args:
            save_name: Report filename

        Creates a comprehensive markdown report with:
        - Experiment configuration
        - Final metrics
        - Links to plots
        - Summary statistics
        """
        report_path = self.exp_dir / save_name

        with open(report_path, 'w') as f:
            # Header
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Configuration
            if self.config:
                f.write("## Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(self.config.to_dict(), indent=2))
                f.write("\n```\n\n")

            # Metrics Summary
            if self.metrics:
                f.write("## Metrics Summary\n\n")
                df = pd.DataFrame(self.metrics)

                # Get final values
                f.write("### Final Values\n\n")
                last_row = df.iloc[-1]
                for col in df.columns:
                    if col not in ['step', 'timestamp']:
                        value = last_row[col]
                        f.write(f"- **{col}**: {value:.4f}\n")

                f.write("\n")

                # Get summary statistics for numeric columns
                f.write("### Statistics\n\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [c for c in numeric_cols if c != 'step']

                if len(numeric_cols) > 0:
                    stats = df[numeric_cols].describe()
                    f.write(stats.to_markdown())
                    f.write("\n\n")

            # Plots
            plots = list(self.plots_dir.glob("*.png"))
            if plots:
                f.write("## Visualizations\n\n")
                for plot in plots:
                    rel_path = plot.relative_to(self.exp_dir)
                    f.write(f"### {plot.stem.replace('_', ' ').title()}\n\n")
                    f.write(f"![{plot.stem}]({rel_path})\n\n")

            # Footer
            f.write("---\n")
            f.write(f"*Report generated automatically by HMT ResultLogger*\n")

        log.info(f"Report generated: {report_path}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment.

        Returns:
            Dictionary with experiment metadata and final metrics
        """
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.exp_dir),
            "num_logged_steps": len(self.metrics),
        }

        if self.config:
            summary["config"] = self.config.to_dict()

        if self.metrics:
            # Add final metrics
            final_metrics = {k: v for k, v in self.metrics[-1].items()
                           if k not in ['step', 'timestamp']}
            summary["final_metrics"] = final_metrics

        return summary


def merge_experiment_results(
    experiment_dirs: List[Path],
    output_file: Path,
    metric_name: str = "perplexity",
):
    """
    Merge results from multiple experiments into single comparison table.

    Useful for creating paper reproduction tables (e.g., Table 1, Table 6).

    Args:
        experiment_dirs: List of experiment directories to merge
        output_file: Path to output CSV file
        metric_name: Metric to extract from each experiment

    Example:
        >>> experiments = [
        ...     Path("results/experiments/exp_001_opt350m"),
        ...     Path("results/experiments/exp_002_opt2.7b"),
        ...     Path("results/experiments/exp_003_llama7b"),
        ... ]
        >>> merge_experiment_results(
        ...     experiments,
        ...     Path("results/paper_reproduction/table1.csv"),
        ...     metric_name="perplexity"
        ... )
    """
    results = []

    for exp_dir in experiment_dirs:
        # Load config
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            log.warning(f"No config found in {exp_dir}")
            continue

        config = ExperimentConfig.load(config_path)

        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            log.warning(f"No metrics found in {exp_dir}")
            continue

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        if not metrics:
            continue

        # Get final metric value
        final_value = metrics[-1].get(metric_name, None)

        if final_value is not None:
            results.append({
                "experiment": config.name,
                "model": config.model_name,
                "dataset": config.dataset,
                metric_name: final_value,
            })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        log.info(f"Merged results saved to {output_file}")
    else:
        log.warning("No results to merge")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("HMT ResultLogger - Local Experiment Tracking")
    print("=" * 80)

    # Create example experiment
    config = ExperimentConfig(
        name="demo_experiment",
        model_name="facebook/opt-350m",
        dataset="wikitext-103",
        segment_length=1024,
        num_memory_embeddings=300,
        sensory_memory_size=32,
        batch_size=2,
        learning_rate=1e-5,
        seed=42,
        paper_table="Table 1",
    )

    result_logger = ResultLogger(
        experiment_name="demo_table1",
        config=config,
        auto_increment=True,
    )

    # Simulate logging metrics
    print("\nLogging metrics...")
    for step in range(0, 500, 50):
        loss = 5.0 * np.exp(-step / 200) + 2.0  # Exponential decay
        ppl = np.exp(loss)
        lr = 1e-5 * (0.9 ** (step // 100))

        result_logger.log_metrics(step=step, loss=loss, perplexity=ppl, learning_rate=lr)

    # Generate plots
    print("\nGenerating plots...")
    result_logger.plot_metrics(["loss", "perplexity"], title="Training Progress")
    result_logger.plot_metrics(["learning_rate"], title="Learning Rate Schedule")

    # Save comparison plot
    print("\nGenerating comparison plot...")
    comparison_data = {
        "HMT": [21.5, 21.3, 21.1, 21.0],
        "Vanilla": [24.5, 25.2, 26.8, 28.5],
        "Sliding Window": [23.1, 23.8, 24.5, 25.2],
    }
    result_logger.plot_comparison(
        comparison_data,
        x_values=[1024, 2048, 4096, 8192],
        xlabel="Sequence Length",
        ylabel="Perplexity",
        title="HMT vs Baselines (Figure 4 Reproduction)",
        save_name="figure4_comparison.png",
    )

    # Save checkpoint
    print("\nSaving checkpoint...")
    dummy_state = {"layer1.weight": np.random.randn(10, 10)}
    result_logger.save_checkpoint(
        dummy_state,
        step=500,
        is_best=True,
        additional_info={"final_ppl": 21.0, "final_loss": 3.04},
    )

    # Generate report
    print("\nGenerating report...")
    result_logger.generate_report()

    # Print summary
    print("\nExperiment Summary:")
    summary = result_logger.get_experiment_summary()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print(f"✅ Demo complete! Check results at: {result_logger.exp_dir}")
    print("=" * 80)
