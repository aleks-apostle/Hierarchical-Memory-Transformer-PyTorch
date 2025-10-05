"""
HMT Training Utilities

Utilities for HMT training including learning rate scheduling, gradient monitoring,
and perplexity calculation.

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

Training utilities based on:
  - Appendix D: Training hyperparameters and configurations
  - Appendix J: Gradient stability analysis
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math
import numpy as np


class ExponentialDecayScheduler:
    """
    Learning rate scheduler with exponential decay.

    Paper Reference:
      - Appendix D, Table 7: "The given learning rate is the starting learning rate
        and will decay by a factor of 0.9 for OPT, OpenLlamaV2, and RWKV models and
        0.7 for the remaining models for every 100 steps."

    The learning rate is updated as:
        lr = initial_lr * (decay_factor ** (step // decay_steps))

    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate
        decay_factor: Decay factor (0.7 or 0.9, from Table 7)
        decay_steps: Apply decay every N steps (default: 100)
        warmup_steps: Number of linear warmup steps (default: 0)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        >>> scheduler = ExponentialDecayScheduler(
        ...     optimizer, initial_lr=1e-5, decay_factor=0.9, decay_steps=100
        ... )
        >>> for step in range(1000):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()  # Update LR every step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        decay_factor: float = 0.9,
        decay_steps: int = 100,
        warmup_steps: int = 0,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Exponential decay
            decay_epochs = (self.current_step - self.warmup_steps) // self.decay_steps
            lr = self.initial_lr * (self.decay_factor**decay_epochs)

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients by global norm.

    Paper Reference:
      - Appendix J: "Gradient Stability in HMT and RMT"
      - Standard practice for BPTT to prevent gradient explosion

    Args:
        model: Model with gradients to clip
        max_norm: Maximum gradient norm (default: 1.0)
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Total norm of gradients before clipping

    Example:
        >>> loss.backward()
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm, norm_type=norm_type
    )
    return total_norm.item()


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for monitoring.

    Paper Reference:
      - Appendix J: Gradient stability analysis
      - Useful for detecting gradient vanishing/explosion

    Args:
        model: Model with gradients

    Returns:
        Dictionary with gradient statistics:
        - grad_norm: L2 norm of all gradients
        - grad_max: Maximum absolute gradient value
        - grad_mean: Mean absolute gradient value
        - grad_std: Standard deviation of gradients

    Example:
        >>> loss.backward()
        >>> stats = compute_gradient_stats(model)
        >>> print(f"Gradient norm: {stats['grad_norm']:.4f}")
    """
    total_norm = 0.0
    all_grads = []

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            all_grads.append(param.grad.data.abs().flatten())

    total_norm = total_norm**0.5

    if len(all_grads) > 0:
        all_grads = torch.cat(all_grads)
        grad_max = all_grads.max().item()
        grad_mean = all_grads.mean().item()
        grad_std = all_grads.std().item()
    else:
        grad_max = 0.0
        grad_mean = 0.0
        grad_std = 0.0

    return {
        "grad_norm": total_norm,
        "grad_max": grad_max,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
    }


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Paper Reference:
      - Perplexity (PPL) is the primary evaluation metric used throughout the paper
      - PPL = exp(loss)
      - Lower PPL indicates better model performance

    Args:
        loss: Cross-entropy loss (scalar tensor or float)

    Returns:
        Perplexity value

    Example:
        >>> loss = F.cross_entropy(logits, targets)
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(loss)


def compute_loss_with_bptt(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for HMT with BPTT.

    Paper Reference:
      - Section 3: HMT processes input in segments
      - Appendix D: Training with BPTT across multiple segments

    Args:
        model: HMT model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] (optional)
        labels: Target token IDs [batch_size, seq_len] (optional, uses input_ids if None)

    Returns:
        Cross-entropy loss (scalar tensor)

    Example:
        >>> loss = compute_loss_with_bptt(hmt_model, input_ids, attention_mask)
        >>> loss.backward()
    """
    # Use input_ids as labels if not provided (standard language modeling)
    if labels is None:
        labels = input_ids.clone()

    # Forward pass through HMT
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    # Get logits
    logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]

    # Shift logits and labels for next-token prediction
    # Logits: predict token at position i using tokens 0..i-1
    # Labels: target is token at position i
    shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]
    shift_labels = labels[:, 1:].contiguous()  # [batch, seq_len-1]

    # Compute cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),  # [batch*(seq_len-1), vocab]
        shift_labels.view(-1),  # [batch*(seq_len-1)]
    )

    return loss


class GradientMonitor:
    """
    Monitor gradient statistics during training.

    Paper Reference:
      - Appendix J: Gradient stability analysis
      - Figure 16-17: Gradient behavior in HMT vs RMT

    This class tracks gradient norms and detects gradient vanishing/explosion.

    Args:
        window_size: Number of steps to track for moving average (default: 100)
        explosion_threshold: Threshold for gradient explosion detection (default: 10.0)
        vanishing_threshold: Threshold for gradient vanishing detection (default: 1e-6)

    Example:
        >>> monitor = GradientMonitor()
        >>> for step in range(1000):
        ...     loss.backward()
        ...     stats = monitor.update(model)
        ...     if stats['is_exploding']:
        ...         print(f"Warning: Gradient explosion at step {step}")
    """

    def __init__(
        self,
        window_size: int = 100,
        explosion_threshold: float = 10.0,
        vanishing_threshold: float = 1e-6,
    ):
        self.window_size = window_size
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.grad_norms: List[float] = []

    def update(self, model: nn.Module) -> Dict[str, float]:
        """
        Update gradient statistics.

        Args:
            model: Model with gradients

        Returns:
            Dictionary with current statistics and alerts
        """
        stats = compute_gradient_stats(model)

        # Track gradient norm history
        self.grad_norms.append(stats["grad_norm"])
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)

        # Compute moving average
        moving_avg = np.mean(self.grad_norms) if len(self.grad_norms) > 0 else 0.0

        # Detect anomalies
        is_exploding = stats["grad_norm"] > self.explosion_threshold
        is_vanishing = stats["grad_norm"] < self.vanishing_threshold

        stats.update(
            {
                "grad_norm_ma": moving_avg,
                "is_exploding": is_exploding,
                "is_vanishing": is_vanishing,
            }
        )

        return stats

    def reset(self):
        """Reset gradient history."""
        self.grad_norms = []


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.

    Paper Reference:
      - Table 7: "EXTRA PARAM" column shows HMT introduces 0.5-1.3% additional parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts:
        - total: Total parameters
        - trainable: Trainable parameters
        - frozen: Frozen parameters

    Example:
        >>> params = count_parameters(hmt_model)
        >>> print(f"Trainable: {params['trainable']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }
