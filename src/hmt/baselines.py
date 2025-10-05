"""
HMT Baseline Comparison Framework

Implements baseline models for comparing against Hierarchical Memory Transformer (HMT).
Provides fair comparison by using the same backbone models with different memory strategies.

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

Baselines Implemented:
1. VanillaTransformer - Standard full-attention baseline (Tables 1, 5, Figure 4)
2. SlidingWindowTransformer - Efficient sliding window attention baseline
3. BaselineEvaluator - Unified comparison interface

These baselines validate HMT's advantages:
- Better long-context performance (Figure 4)
- Higher efficiency (Table 5: 1.5-2.4× speedup)
- Superior perplexity (Table 1)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .model import HMT
from .evaluation import HMTEvaluator
from .results import ResultLogger

log = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    model_type: str  # 'vanilla' or 'sliding_window'
    backbone_name: str  # HuggingFace model name
    max_length: Optional[int] = None  # Maximum sequence length (None = no limit)

    # Sliding window specific
    window_size: int = 512  # Size of attention window
    stride: int = 256  # Overlap between windows

    # Device
    device: Optional[str] = None  # Auto-detected if None


class VanillaTransformer(nn.Module):
    """
    Vanilla Transformer Baseline - Standard full-attention model.

    Paper Reference:
      - Tables 1, 5-7: "Vanilla" baseline column
      - Figure 4: Shows vanilla degrading on long contexts
      - Section 4: Baseline comparison methodology

    Architecture:
      - Uses backbone model directly without any memory augmentation
      - Full O(L²) attention complexity over entire sequence
      - Limited by backbone's positional encoding (typically 1024-2048 tokens)

    Limitations:
      - Quadratic complexity: Slow on long sequences
      - Out-of-distribution for lengths > training length
      - No explicit long-term memory mechanism

    Truncation Strategies:
      - 'head': Keep first max_length tokens (loses recent context)
      - 'tail': Keep last max_length tokens (loses beginning)
      - 'middle': Keep middle max_length tokens (loses both ends)

    This baseline demonstrates the problem HMT solves: handling long contexts
    efficiently while maintaining performance.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> backbone = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> vanilla = VanillaTransformer(backbone, max_length=1024)
        >>> outputs = vanilla(input_ids)  # Works up to 1024 tokens
        >>> # Beyond 1024: either truncates or OOMs
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        max_length: Optional[int] = None,
        truncation_strategy: str = 'tail',
    ):
        """
        Initialize Vanilla Transformer baseline.

        Args:
            backbone_model: Pre-trained transformer model (e.g., GPT-2, OPT)
            max_length: Maximum sequence length (None = use backbone's limit)
            truncation_strategy: How to truncate long sequences ('head', 'tail', 'middle')
        """
        super().__init__()

        self.backbone_model = backbone_model
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

        # Get backbone's actual max length from config
        self.backbone_max_length = getattr(
            backbone_model.config,
            'max_position_embeddings',
            2048
        )

        if self.max_length is None:
            self.max_length = self.backbone_max_length

        log.info(f"VanillaTransformer initialized: max_length={self.max_length}, "
                f"backbone_max={self.backbone_max_length}, strategy={truncation_strategy}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional truncation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_dict: Whether to return dictionary

        Returns:
            Dictionary with 'logits' key: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Truncate if necessary
        if seq_len > self.max_length:
            input_ids, attention_mask = self._truncate(
                input_ids,
                attention_mask,
                self.max_length
            )
            log.debug(f"Truncated sequence from {seq_len} to {self.max_length} tokens")

        # Standard forward pass through backbone
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        if return_dict:
            return {
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            }
        else:
            return outputs.logits

    def _truncate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Truncate sequences according to strategy.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] or None
            max_length: Target length

        Returns:
            Truncated (input_ids, attention_mask)
        """
        seq_len = input_ids.size(1)

        if self.truncation_strategy == 'tail':
            # Keep last max_length tokens (most recent context)
            input_ids = input_ids[:, -max_length:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_length:]

        elif self.truncation_strategy == 'head':
            # Keep first max_length tokens (beginning of document)
            input_ids = input_ids[:, :max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_length]

        elif self.truncation_strategy == 'middle':
            # Keep middle max_length tokens
            start = (seq_len - max_length) // 2
            end = start + max_length
            input_ids = input_ids[:, start:end]
            if attention_mask is not None:
                attention_mask = attention_mask[:, start:end]

        else:
            raise ValueError(f"Unknown truncation strategy: {self.truncation_strategy}")

        return input_ids, attention_mask


class SlidingWindowTransformer(nn.Module):
    """
    Sliding Window Transformer Baseline - Efficient chunked attention.

    Paper Reference:
      - Common long-context baseline for comparison
      - More efficient than vanilla (O(L) vs O(L²))
      - But loses global context unlike HMT

    Architecture:
      - Splits long sequence into overlapping windows
      - Processes each window independently
      - Aggregates outputs with stride-based stitching

    Parameters:
      - window_size: Size of each attention window (e.g., 512)
      - stride: Overlap between windows (e.g., 256)
        * Higher stride = more overlap = better context but slower
        * Lower stride = less overlap = faster but may miss dependencies

    Advantages over Vanilla:
      - Linear O(L) complexity in sequence length
      - Can process arbitrarily long sequences
      - No truncation needed

    Limitations vs HMT:
      - No explicit long-term memory retrieval
      - Loses information between non-overlapping windows
      - Cannot attend to distant context beyond window

    This baseline shows that efficiency alone isn't enough - HMT's memory
    retrieval mechanism provides both efficiency AND long-range understanding.

    Example:
        >>> backbone = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> sliding = SlidingWindowTransformer(
        ...     backbone, window_size=512, stride=256
        ... )
        >>> outputs = sliding(very_long_input)  # Can handle 10K+ tokens
        >>> # Each window sees 512 tokens, with 256 token overlap
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        window_size: int = 512,
        stride: int = 256,
    ):
        """
        Initialize Sliding Window Transformer.

        Args:
            backbone_model: Pre-trained transformer
            window_size: Size of each attention window
            stride: Number of tokens to advance between windows
        """
        super().__init__()

        self.backbone_model = backbone_model
        self.window_size = window_size
        self.stride = stride

        assert stride <= window_size, "Stride must be <= window_size for overlap"

        log.info(f"SlidingWindowTransformer initialized: window={window_size}, stride={stride}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sliding window attention.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_dict: Whether to return dict

        Returns:
            Dictionary with 'logits': [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # If sequence fits in one window, use standard forward pass
        if seq_len <= self.window_size:
            outputs = self.backbone_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )

            if return_dict:
                return {'logits': outputs.logits}
            return outputs.logits

        # Split into overlapping windows
        windows = self._create_windows(input_ids, attention_mask)

        # Process each window
        all_logits = []

        for window_input_ids, window_mask in windows:
            outputs = self.backbone_model(
                input_ids=window_input_ids,
                attention_mask=window_mask,
                return_dict=True,
                **kwargs,
            )
            all_logits.append(outputs.logits)

        # Stitch windows together
        stitched_logits = self._stitch_windows(all_logits)

        if return_dict:
            return {'logits': stitched_logits}
        return stitched_logits

    def _create_windows(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create overlapping windows from sequence.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] or None

        Returns:
            List of (window_input_ids, window_mask) tuples
        """
        batch_size, seq_len = input_ids.shape
        windows = []

        start = 0
        while start < seq_len:
            end = min(start + self.window_size, seq_len)

            window_input_ids = input_ids[:, start:end]
            window_mask = attention_mask[:, start:end] if attention_mask is not None else None

            # Pad last window if necessary
            if window_input_ids.size(1) < self.window_size:
                pad_length = self.window_size - window_input_ids.size(1)
                window_input_ids = torch.nn.functional.pad(
                    window_input_ids, (0, pad_length), value=0
                )
                if window_mask is not None:
                    window_mask = torch.nn.functional.pad(
                        window_mask, (0, pad_length), value=0
                    )

            windows.append((window_input_ids, window_mask))

            start += self.stride

        return windows

    def _stitch_windows(self, window_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Stitch overlapping window outputs together.

        Strategy: For overlapping regions, average the logits from both windows.

        Args:
            window_logits: List of [batch, window_size, vocab_size] tensors

        Returns:
            Stitched logits [batch, total_seq_len, vocab_size]
        """
        if len(window_logits) == 1:
            return window_logits[0]

        batch_size, _, vocab_size = window_logits[0].shape

        # Calculate total sequence length
        num_windows = len(window_logits)
        total_length = (num_windows - 1) * self.stride + self.window_size

        # Initialize output tensor and count tensor for averaging
        stitched = torch.zeros(
            batch_size, total_length, vocab_size,
            device=window_logits[0].device,
            dtype=window_logits[0].dtype
        )
        counts = torch.zeros(batch_size, total_length, 1, device=stitched.device)

        # Add each window's contribution
        for i, logits in enumerate(window_logits):
            start = i * self.stride
            end = start + self.window_size

            # Handle last window which might be padded
            actual_length = min(self.window_size, total_length - start)

            stitched[:, start:start+actual_length, :] += logits[:, :actual_length, :]
            counts[:, start:start+actual_length, :] += 1

        # Average overlapping regions
        stitched = stitched / counts.clamp(min=1)

        return stitched


class BaselineEvaluator:
    """
    Unified evaluator for comparing HMT against baselines.

    Paper Reference:
      - Tables 1, 5-7: Direct comparisons between HMT and baselines
      - Figure 4: Long-context performance comparison
      - Section 4: Experimental methodology

    This class provides a unified interface for:
    1. Evaluating multiple models on same datasets
    2. Generating comparison tables and figures
    3. Reproducing paper results

    Models supported:
      - HMT (with memory hierarchy)
      - VanillaTransformer (full attention)
      - SlidingWindowTransformer (efficient chunking)
      - Custom baselines (extensible)

    Example Usage:
        >>> # Initialize models
        >>> hmt_model = HMT(backbone, config)
        >>> vanilla_model = VanillaTransformer(backbone)
        >>> sliding_model = SlidingWindowTransformer(backbone, window_size=512)
        >>>
        >>> # Create evaluator
        >>> evaluator = BaselineEvaluator(
        ...     models={
        ...         'HMT': hmt_model,
        ...         'Vanilla': vanilla_model,
        ...         'Sliding Window': sliding_model,
        ...     }
        ... )
        >>>
        >>> # Compare on dataset
        >>> results = evaluator.evaluate_all_models(test_loader)
        >>> print(results)  # {'HMT': {'ppl': 21.3}, 'Vanilla': {'ppl': 24.1}, ...}
        >>>
        >>> # Reproduce Figure 4
        >>> fig4_results = evaluator.compare_long_context(
        ...     test_dataset,
        ...     lengths=[512, 1024, 2048, 4096, 8192]
        ... )
        >>>
        >>> # Reproduce Table 5
        >>> speed_results = evaluator.compare_efficiency(test_loader)
        >>> print(f"HMT speedup: {speed_results['speedup']['HMT']:.2f}x")
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: Optional[str] = None,
        result_logger: Optional[ResultLogger] = None,
    ):
        """
        Initialize baseline evaluator.

        Args:
            models: Dictionary mapping model names to model instances
            device: Device to use (auto-detected if None)
            result_logger: Optional logger for saving results
        """
        self.models = models
        self.device = device if device is not None else self._get_device()
        self.result_logger = result_logger

        # Move all models to device
        for name, model in self.models.items():
            model.to(self.device)
            log.info(f"Loaded model '{name}' on {self.device}")

    def _get_device(self) -> str:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @torch.no_grad()
    def evaluate_all_models(
        self,
        dataloader,
        metric: str = 'perplexity',
        max_batches: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on the same dataset.

        Args:
            dataloader: DataLoader with evaluation data
            metric: Metric to compute ('perplexity', 'accuracy', etc.)
            max_batches: Maximum batches to evaluate (None = all)

        Returns:
            Dictionary mapping model_name -> {metric_name: value}

        Example:
            >>> results = evaluator.evaluate_all_models(test_loader)
            >>> # {'HMT': {'perplexity': 21.3, 'loss': 3.06},
            >>> #  'Vanilla': {'perplexity': 24.1, 'loss': 3.18}, ...}
        """
        results = {}

        for model_name, model in self.models.items():
            log.info(f"Evaluating {model_name}...")

            model.eval()
            total_loss = 0.0
            total_tokens = 0

            pbar = tqdm(dataloader, desc=f"{model_name}")
            for batch_idx, batch in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                # Compute loss
                logits = outputs['logits']
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Count tokens
                if attention_mask is not None:
                    num_tokens = attention_mask[:, 1:].sum().item()
                else:
                    num_tokens = shift_labels.numel()

                total_loss += loss.item()
                total_tokens += num_tokens

                # Clear memory if HMT
                if hasattr(model, 'clear_memory'):
                    model.clear_memory()

                # Update progress
                current_ppl = np.exp(total_loss / total_tokens)
                pbar.set_postfix({'ppl': f'{current_ppl:.2f}'})

            # Compute final metrics
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)

            results[model_name] = {
                'perplexity': float(perplexity),
                'loss': float(avg_loss),
                'num_tokens': int(total_tokens),
            }

            log.info(f"{model_name}: PPL={perplexity:.2f}, Loss={avg_loss:.4f}")

        return results

    @torch.no_grad()
    def compare_long_context(
        self,
        dataset,
        sequence_lengths: List[int] = [512, 1024, 2048, 4096, 8192],
        num_samples: int = 100,
    ) -> Dict[str, Dict[int, float]]:
        """
        Compare models at different sequence lengths (Figure 4 reproduction).

        Paper Reference:
          - Figure 4: "Extrapolation to Longer Context"
          - Shows HMT maintains performance while baselines degrade

        Args:
            dataset: Dataset to sample from
            sequence_lengths: List of lengths to test
            num_samples: Samples per length

        Returns:
            Dictionary: {model_name: {length: perplexity}}

        Example:
            >>> results = evaluator.compare_long_context(
            ...     test_dataset,
            ...     lengths=[512, 1024, 2048, 4096, 8192]
            ... )
            >>> # Plot Figure 4
            >>> for model_name, length_ppls in results.items():
            ...     plt.plot(list(length_ppls.keys()), list(length_ppls.values()),
            ...              label=model_name, marker='o')
            >>> plt.xlabel("Sequence Length")
            >>> plt.ylabel("Perplexity")
            >>> plt.title("Long-Context Extrapolation (Figure 4)")
            >>> plt.legend()
        """
        results = {name: {} for name in self.models.keys()}

        for seq_len in sequence_lengths:
            log.info(f"Evaluating at sequence length: {seq_len}")

            for model_name, model in self.models.items():
                model.eval()

                total_loss = 0.0
                total_tokens = 0
                num_evaluated = 0

                for idx in tqdm(range(min(num_samples, len(dataset))),
                               desc=f"{model_name} @ {seq_len}"):
                    sample = dataset[idx % len(dataset)]
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)

                    # Truncate/pad to target length
                    if input_ids.size(1) > seq_len:
                        input_ids = input_ids[:, :seq_len]
                    elif input_ids.size(1) < seq_len:
                        continue

                    # Forward pass
                    outputs = model(input_ids=input_ids, return_dict=True)

                    # Compute loss
                    logits = outputs['logits']
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss(reduction='sum')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    total_loss += loss.item()
                    total_tokens += shift_labels.numel()
                    num_evaluated += 1

                    if hasattr(model, 'clear_memory'):
                        model.clear_memory()

                # Compute PPL for this length
                if total_tokens > 0:
                    ppl = np.exp(total_loss / total_tokens)
                    results[model_name][seq_len] = float(ppl)
                    log.info(f"  {model_name} @ {seq_len}: PPL={ppl:.2f}")

        return results

    @torch.no_grad()
    def compare_efficiency(
        self,
        dataloader,
        num_warmup: int = 5,
        num_measure: int = 50,
    ) -> Dict[str, Any]:
        """
        Compare inference speed across models (Table 5 reproduction).

        Paper Reference:
          - Table 5: "Inference Time Comparison"
          - HMT achieves 1.5-2.4× speedup over vanilla

        Args:
            dataloader: Benchmark dataloader
            num_warmup: Warmup iterations
            num_measure: Measurement iterations

        Returns:
            Dictionary with throughput, latency, and speedup metrics
        """
        import time

        results = {}

        for model_name, model in self.models.items():
            log.info(f"Benchmarking {model_name}...")
            model.eval()

            # Warmup
            dataloader_iter = iter(dataloader)
            for _ in range(num_warmup):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                input_ids = batch['input_ids'].to(self.device)
                _ = model(input_ids, return_dict=True)
                if hasattr(model, 'clear_memory'):
                    model.clear_memory()

            # Measure
            total_time = 0.0
            total_tokens = 0

            for _ in tqdm(range(num_measure), desc=f"Bench {model_name}"):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                input_ids = batch['input_ids'].to(self.device)
                num_tokens = input_ids.numel()

                start = time.perf_counter()
                _ = model(input_ids, return_dict=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif torch.backends.mps.is_available():
                    torch.mps.synchronize()

                end = time.perf_counter()

                total_time += (end - start)
                total_tokens += num_tokens

                if hasattr(model, 'clear_memory'):
                    model.clear_memory()

            throughput = total_tokens / total_time
            latency = total_time / num_measure

            results[model_name] = {
                'tokens_per_second': float(throughput),
                'seconds_per_batch': float(latency),
                'total_tokens': int(total_tokens),
            }

            log.info(f"{model_name}: {throughput:.1f} tok/s, {latency*1000:.1f} ms/batch")

        # Compute speedup relative to vanilla or slowest model
        baseline_speed = results.get('Vanilla', results[list(results.keys())[0]])['tokens_per_second']

        speedups = {}
        for model_name, metrics in results.items():
            speedups[model_name] = metrics['tokens_per_second'] / baseline_speed

        results['speedup'] = speedups

        return results

    def generate_comparison_report(
        self,
        results: Dict[str, Any],
        output_path: Path,
        title: str = "Model Comparison Report",
    ):
        """
        Generate comprehensive comparison report.

        Args:
            results: Results from evaluate_all_models or other methods
            output_path: Path to save markdown report
            title: Report title
        """
        with open(output_path, 'w') as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Generated:** {Path(__file__).name}\n\n")

            f.write("## Model Comparison\n\n")
            f.write("| Model | Perplexity | Loss | Speedup |\n")
            f.write("|-------|------------|------|---------|\n")

            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and 'perplexity' in metrics:
                    ppl = metrics.get('perplexity', '-')
                    loss = metrics.get('loss', '-')
                    speedup = results.get('speedup', {}).get(model_name, '-')
                    f.write(f"| {model_name} | {ppl:.2f} | {loss:.4f} | {speedup:.2f}× |\n")

            f.write("\n---\n")
            f.write("*Generated by BaselineEvaluator*\n")

        log.info(f"Comparison report saved: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("HMT Baseline Comparison Framework")
    print("=" * 80)

    print("\nAvailable baseline models:")
    print("1. VanillaTransformer - Standard full-attention (O(L²))")
    print("2. SlidingWindowTransformer - Efficient windowed attention (O(L))")
    print("3. BaselineEvaluator - Unified comparison interface")

    print("\nUsage example:")
    print("""
    from transformers import AutoModelForCausalLM
    from hmt.baselines import VanillaTransformer, SlidingWindowTransformer, BaselineEvaluator

    # Load backbone
    backbone = AutoModelForCausalLM.from_pretrained('gpt2')

    # Create baselines
    vanilla = VanillaTransformer(backbone, max_length=1024)
    sliding = SlidingWindowTransformer(backbone, window_size=512, stride=256)

    # Compare with HMT
    evaluator = BaselineEvaluator({
        'HMT': hmt_model,
        'Vanilla': vanilla,
        'Sliding Window': sliding,
    })

    results = evaluator.evaluate_all_models(test_loader)
    """)

    print("\n" + "=" * 80)
