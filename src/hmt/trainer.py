"""
HMT Trainer with Multi-Stage BPTT

Main trainer class for training HMT models with Backpropagation Through Time (BPTT).

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

Implementation based on:
  - Appendix F: Multi-stage Training
  - Appendix D: Training Details and Hyperparameters
  - Appendix J: Gradient Stability in HMT
  - Algorithm 1: HMT training procedure
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import json
import logging

from .training_config import TrainingConfig
from .training_utils import (
    ExponentialDecayScheduler,
    clip_gradients,
    compute_gradient_stats,
    compute_perplexity,
    compute_loss_with_bptt,
    GradientMonitor,
    set_seed,
    count_parameters,
)
from .model import HMT
from .config import HMTConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HMTTrainer:
    """
    Trainer for HMT models with multi-stage BPTT.

    Paper Reference:
      - Appendix F: "Multi-stage Training"
        "Training HMT involves multiple segments of tokens to learn how to encode
        input tokens and retrieve information properly. Therefore, we split training
        HMT into two stages."

      Stage 1 (Lines 1-8, Appendix F):
        - Train WITHOUT memory retrieval mechanism
        - BPTT with 2 segments unrolled
        - Simpler architecture, faster iterations (~1.15 s/iteration)
        - Purpose: Learn basic segment encoding and memory generation

      Stage 2 (Lines 9-15, Appendix F):
        - Load checkpoint from Stage 1 and add memory retrieval
        - BPTT with maximum segments GPU can handle (15 in paper)
        - Full HMT architecture (~3.36 s/iteration)
        - Purpose: Learn memory retrieval and long-range dependencies

    Args:
        model: HMT model to train
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader (optional)
        optimizer: Optimizer (optional, will create AdamW if None)

    Example:
        >>> config = TrainingConfig(
        ...     backbone_model_name="facebook/opt-350m",
        ...     use_multi_stage=True,
        ...     stage1_steps=200,
        ...     stage2_steps=500,
        ... )
        >>> trainer = HMTTrainer(model, config, train_loader)
        >>> trainer.train()  # Runs both stages automatically
    """

    def __init__(
        self,
        model: HMT,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Move model to device
        self.device = config.device
        self.model = self.model.to(self.device)

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Create learning rate scheduler (Appendix D, Table 7)
        self.scheduler = ExponentialDecayScheduler(
            self.optimizer,
            initial_lr=config.learning_rate,
            decay_factor=config.lr_decay_factor,
            decay_steps=config.lr_decay_steps,
            warmup_steps=config.warmup_steps,
        )

        # Gradient monitor for stability tracking (Appendix J)
        self.grad_monitor = GradientMonitor()

        # Training state
        self.global_step = 0
        self.current_stage = 0  # 0 = not started, 1 = stage 1, 2 = stage 2
        self.best_eval_metric = float("inf")

        # W&B logging
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict(),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B logging.")
                self.use_wandb = False

        # Log model info
        param_counts = count_parameters(self.model)
        logger.info(f"Model parameters: {param_counts['trainable']:,} trainable")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training config: {config.to_dict()}")

    def train(self) -> Dict[str, Any]:
        """
        Run complete training (both stages if multi-stage is enabled).

        Paper Reference:
          - Appendix F: Multi-stage training methodology
          - Figure 12: Performance comparison showing multi-stage is better

        Returns:
            Dictionary with training statistics

        Example:
            >>> stats = trainer.train()
            >>> print(f"Final perplexity: {stats['final_perplexity']:.2f}")
        """
        if self.config.use_multi_stage:
            logger.info("=" * 80)
            logger.info("Multi-Stage Training (Appendix F)")
            logger.info("=" * 80)

            # === Stage 1: Train without memory retrieval ===
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1: Training WITHOUT memory retrieval")
            logger.info(f"  - BPTT segments: {self.config.stage1_bptt_segments}")
            logger.info(f"  - Steps: {self.config.stage1_steps}")
            logger.info(f"  - Purpose: Learn basic segment encoding")
            logger.info("=" * 80 + "\n")

            stage1_stats = self.train_stage(
                stage=1,
                num_steps=self.config.stage1_steps,
                bptt_segments=self.config.stage1_bptt_segments,
                use_memory=False,  # Disable memory retrieval in Stage 1
            )

            # Save Stage 1 checkpoint
            stage1_ckpt = self.config.output_dir / "stage1_final.pt"
            self.save_checkpoint(stage1_ckpt)
            logger.info(f"Stage 1 checkpoint saved to {stage1_ckpt}")

            # === Stage 2: Train with memory retrieval ===
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: Training WITH memory retrieval")
            logger.info(f"  - BPTT segments: {self.config.stage2_bptt_segments}")
            logger.info(f"  - Steps: {self.config.stage2_steps}")
            logger.info(f"  - Purpose: Learn memory retrieval and long-range dependencies")
            logger.info("=" * 80 + "\n")

            stage2_stats = self.train_stage(
                stage=2,
                num_steps=self.config.stage2_steps,
                bptt_segments=self.config.stage2_bptt_segments,
                use_memory=True,  # Enable memory retrieval in Stage 2
            )

            # Combine statistics
            stats = {
                "stage1": stage1_stats,
                "stage2": stage2_stats,
                "final_perplexity": stage2_stats["final_perplexity"],
                "best_eval_metric": self.best_eval_metric,
            }

        else:
            # Single-stage training (standard BPTT)
            logger.info("Single-Stage Training")
            stats = self.train_stage(
                stage=1,
                num_steps=self.config.stage2_steps,
                bptt_segments=self.config.bptt_unroll_depth,
                use_memory=True,
            )

        # Save final checkpoint
        final_ckpt = self.config.output_dir / "final.pt"
        self.save_checkpoint(final_ckpt)
        logger.info(f"Final checkpoint saved to {final_ckpt}")

        if self.use_wandb:
            self.wandb.finish()

        return stats

    def train_stage(
        self,
        stage: int,
        num_steps: int,
        bptt_segments: int,
        use_memory: bool,
    ) -> Dict[str, Any]:
        """
        Train a single stage with BPTT.

        Paper Reference:
          - Appendix D: "Training configurations"
          - Appendix J: "Gradient Stability" - HMT can handle deeper BPTT unroll

        Args:
            stage: Stage number (1 or 2)
            num_steps: Number of training steps
            bptt_segments: Number of segments to unroll in BPTT
            use_memory: Whether to use memory retrieval mechanism

        Returns:
            Dictionary with stage statistics
        """
        self.current_stage = stage
        self.model.train()

        # Initialize tracking
        total_loss = 0.0
        total_perplexity = 0.0
        step_in_stage = 0

        # Create progress bar
        pbar = tqdm(total=num_steps, desc=f"Stage {stage}")

        # Training loop
        train_iter = iter(self.train_dataloader)
        while step_in_stage < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator if we've exhausted the dataset
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # === BPTT Forward and Backward Pass ===
            # Paper Reference: Appendix D, Training with BPTT
            loss = self._bptt_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bptt_segments=bptt_segments,
                use_memory=use_memory,
            )

            # === Gradient Clipping (Standard BPTT practice) ===
            grad_norm = clip_gradients(self.model, max_norm=self.config.gradient_clip_norm)

            # === Optimizer Step ===
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Clear HMT memory between batches (different documents)
            self.model.clear_memory()

            # === Tracking ===
            step_in_stage += 1
            self.global_step += 1
            total_loss += loss
            ppl = compute_perplexity(loss)
            total_perplexity += ppl

            # === Logging ===
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / step_in_stage
                avg_ppl = total_perplexity / step_in_stage
                lr = self.scheduler.get_last_lr()

                # Compute gradient statistics (Appendix J)
                grad_stats = self.grad_monitor.update(self.model)

                log_dict = {
                    "stage": stage,
                    "step": self.global_step,
                    "loss": loss,
                    "perplexity": ppl,
                    "avg_loss": avg_loss,
                    "avg_perplexity": avg_ppl,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                    "grad_norm_ma": grad_stats["grad_norm_ma"],
                }

                # Log to W&B
                if self.use_wandb:
                    self.wandb.log(log_dict, step=self.global_step)

                # Update progress bar
                pbar.set_postfix(
                    {"loss": f"{loss:.4f}", "ppl": f"{ppl:.2f}", "lr": f"{lr:.2e}"}
                )

                # Check for gradient issues (Appendix J)
                if grad_stats["is_exploding"]:
                    logger.warning(
                        f"Gradient explosion detected! Norm: {grad_stats['grad_norm']:.2f}"
                    )
                if grad_stats["is_vanishing"]:
                    logger.warning(
                        f"Gradient vanishing detected! Norm: {grad_stats['grad_norm']:.2e}"
                    )

            # === Evaluation ===
            if (
                self.eval_dataloader is not None
                and self.global_step % self.config.eval_steps == 0
            ):
                eval_metrics = self.evaluate()
                logger.info(
                    f"Step {self.global_step}: Eval {self.config.eval_metric} = {eval_metrics[self.config.eval_metric]:.4f}"
                )

                if self.use_wandb:
                    self.wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()}, step=self.global_step)

                # Save best model
                if eval_metrics[self.config.eval_metric] < self.best_eval_metric:
                    self.best_eval_metric = eval_metrics[self.config.eval_metric]
                    best_ckpt = self.config.output_dir / f"best_stage{stage}.pt"
                    self.save_checkpoint(best_ckpt)
                    logger.info(f"New best model saved: {self.config.eval_metric} = {self.best_eval_metric:.4f}")

                self.model.train()

            # === Checkpointing ===
            if self.global_step % self.config.save_steps == 0:
                ckpt_path = self.config.output_dir / f"checkpoint_step{self.global_step}.pt"
                self.save_checkpoint(ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

            pbar.update(1)

        pbar.close()

        # Stage statistics
        avg_loss = total_loss / num_steps
        avg_ppl = total_perplexity / num_steps

        stats = {
            "num_steps": num_steps,
            "final_loss": loss,
            "final_perplexity": ppl,
            "avg_loss": avg_loss,
            "avg_perplexity": avg_ppl,
        }

        logger.info(f"\nStage {stage} Complete:")
        logger.info(f"  Final Loss: {loss:.4f}")
        logger.info(f"  Final Perplexity: {ppl:.2f}")
        logger.info(f"  Avg Perplexity: {avg_ppl:.2f}")

        return stats

    def _bptt_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        bptt_segments: int,
        use_memory: bool,
    ) -> float:
        """
        Perform one BPTT step with unrolled segments.

        Paper Reference:
          - Appendix D: "BPTT unroll depth"
          - Appendix F: Multi-stage training uses different unroll depths

        This method:
        1. Segments the input into chunks
        2. Unrolls BPTT for K segments
        3. Accumulates gradients across segments
        4. Returns average loss

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            bptt_segments: Number of segments to unroll
            use_memory: Whether to use memory retrieval

        Returns:
            Average loss across segments
        """
        batch_size, seq_len = input_ids.shape
        segment_length = self.model.config.segment_length

        # Calculate how many segments we can process
        num_segments = min(bptt_segments, (seq_len + segment_length - 1) // segment_length)

        # Truncate input to fit exact number of segments
        truncated_len = num_segments * segment_length
        if seq_len > truncated_len:
            input_ids = input_ids[:, :truncated_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :truncated_len]

        # === Forward pass through HMT ===
        # HMT.forward() handles segmentation internally
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_memory=use_memory,
            return_dict=True,
        )

        # === Compute loss ===
        loss = compute_loss_with_bptt(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # === Backward pass ===
        loss.backward()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation dataset.

        Returns:
            Dictionary with evaluation metrics (loss, perplexity)
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Compute loss
            loss = compute_loss_with_bptt(self.model, input_ids, attention_mask)

            total_loss += loss.item()
            num_batches += 1

            # Clear memory between batches
            self.model.clear_memory()

        # Compute metrics
        avg_loss = total_loss / num_batches
        perplexity = compute_perplexity(avg_loss)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, path: Path):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "best_eval_metric": self.best_eval_metric,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.current_step = checkpoint["scheduler_step"]
        self.global_step = checkpoint["global_step"]
        self.current_stage = checkpoint["current_stage"]
        self.best_eval_metric = checkpoint["best_eval_metric"]

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"  Global step: {self.global_step}")
        logger.info(f"  Current stage: {self.current_stage}")
