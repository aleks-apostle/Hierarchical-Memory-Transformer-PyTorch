"""
Hierarchical Memory Transformer (HMT)
A memory-augmented framework for efficient long-context language processing.
"""

__version__ = "0.1.0"

from .config import HMTConfig
from .model import HMT
from .utils import get_device
from .data import WikiTextDataset, LongContextDataLoader, create_dataloaders
from .trainer import HMTTrainer
from .training_config import TrainingConfig
from .evaluation import HMTEvaluator, compute_perplexity, compute_bits_per_byte
from .results import ResultLogger, ExperimentConfig
from .benchmarks import PG19Dataset, ArXivDataset, LongContextBenchmark, create_benchmark_dataloaders
from .baselines import VanillaTransformer, SlidingWindowTransformer, BaselineEvaluator, BaselineConfig

__all__ = [
    "HMT",
    "HMTConfig",
    "get_device",
    "WikiTextDataset",
    "LongContextDataLoader",
    "create_dataloaders",
    "HMTTrainer",
    "TrainingConfig",
    "HMTEvaluator",
    "compute_perplexity",
    "compute_bits_per_byte",
    "ResultLogger",
    "ExperimentConfig",
    "PG19Dataset",
    "ArXivDataset",
    "LongContextBenchmark",
    "create_benchmark_dataloaders",
    "VanillaTransformer",
    "SlidingWindowTransformer",
    "BaselineEvaluator",
    "BaselineConfig",
]
