"""
HMT Training Configuration

Configuration dataclass for HMT training with BPTT (Backpropagation Through Time).

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

Training details from:
  - Appendix D: HMT, RMT, and Baseline Training Details and Hyperparameters
  - Table 7: Training and fine-tuning configurations
  - Appendix F: Multi-stage Training
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for HMT training with BPTT.

    Paper References:
      - Table 7 (Appendix D): Complete training configurations for all models
      - Appendix F: Multi-stage training methodology
      - Appendix J: Gradient stability and BPTT unroll depth

    Multi-Stage Training (Appendix F):
      Stage 1: Train without memory retrieval mechanism
               - Shorter sequences (2 segments)
               - Simpler architecture
               - Faster iterations (~1.15 s/iter)
               - 200 steps default

      Stage 2: Train with full memory retrieval mechanism
               - Longer sequences (maximum GPU can handle)
               - Complete HMT architecture
               - 500 steps default

    Attributes:
        # === Model Configuration ===
        backbone_model_name: HuggingFace model name or path
        hmt_config_path: Path to HMTConfig (optional, will create default)

        # === Training Stages (Appendix F) ===
        use_multi_stage: Whether to use 2-stage training (recommended)
        stage1_steps: Training steps for Stage 1 (default: 200)
        stage2_steps: Training steps for Stage 2 (default: 500)
        stage1_bptt_segments: Segments to unroll in Stage 1 (default: 2)
        stage2_bptt_segments: Segments to unroll in Stage 2 (default: 15)

        # === BPTT Configuration (Appendix D, J) ===
        bptt_unroll_depth: Number of segments to unroll for single-stage training
        truncated_bptt: Whether to truncate gradients between segments

        # === Optimization (Table 7) ===
        learning_rate: Initial learning rate (1e-5 to 2e-4 based on model)
        lr_decay_factor: LR decay factor (0.7 or 0.9, applied every 100 steps)
        lr_decay_steps: Apply LR decay every N steps (default: 100)
        batch_size: Batch size (default: 2, Table 7)
        gradient_clip_norm: Max gradient norm for clipping (default: 1.0)
        weight_decay: L2 regularization (default: 0.01)
        warmup_steps: Number of warmup steps (default: 0)

        # === Training Data (Appendix D) ===
        dataset_name: Dataset for training (default: "togethercomputer/RedPajama-Data-V2")
        max_length: Maximum sequence length for training
        num_workers: DataLoader workers (default: 4)
        shuffle: Whether to shuffle training data (default: True)

        # === Checkpointing and Logging ===
        output_dir: Directory for checkpoints and logs
        save_steps: Save checkpoint every N steps (default: 100)
        eval_steps: Evaluate every N steps (default: 100)
        logging_steps: Log metrics every N steps (default: 10)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)

        # === W&B Logging ===
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional, auto-generated if None)

        # === Memory Optimization (Appendix K) ===
        use_deepspeed: Enable DeepSpeed ZeRO Stage 2 (recommended for large models)
        use_lora: Use LoRA for parameter-efficient training (rank 8, for 7B+ models)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        mixed_precision: Use mixed precision training ("fp16", "bf16", or None)
        cpu_offload: Offload intermediate data to CPU (Appendix K)

        # === Evaluation ===
        eval_dataset_name: Dataset for evaluation (default: "Salesforce/wikitext")
        eval_max_length: Maximum sequence length for evaluation
        eval_metric: Metric to monitor ("perplexity" or "loss")

        # === Device ===
        device: Device to use (auto-detected if None)
        num_gpus: Number of GPUs to use (default: 1)

        # === Reproducibility ===
        seed: Random seed for reproducibility (default: 42)
    """

    # === Model Configuration ===
    backbone_model_name: str = "facebook/opt-350m"
    hmt_config_path: Optional[Path] = None

    # === Training Stages (Appendix F) ===
    use_multi_stage: bool = True
    stage1_steps: int = 200
    stage2_steps: int = 500
    stage1_bptt_segments: int = 2
    stage2_bptt_segments: int = 15

    # === BPTT Configuration ===
    bptt_unroll_depth: int = 4
    truncated_bptt: bool = False

    # === Optimization (Table 7, Appendix D) ===
    learning_rate: float = 1e-5
    lr_decay_factor: float = 0.9  # 0.9 for OPT/OpenLlama/RWKV, 0.7 for others
    lr_decay_steps: int = 100
    batch_size: int = 2
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 0

    # === Training Data ===
    dataset_name: str = "togethercomputer/RedPajama-Data-V2"
    max_length: int = 2048
    num_workers: int = 4
    shuffle: bool = True

    # === Checkpointing and Logging ===
    output_dir: str = "./checkpoints"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3

    # === W&B Logging ===
    use_wandb: bool = False
    wandb_project: str = "hmt-training"
    wandb_run_name: Optional[str] = None

    # === Memory Optimization (Appendix K) ===
    use_deepspeed: bool = False
    use_lora: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: Optional[Literal["fp16", "bf16"]] = None
    cpu_offload: bool = False

    # === Evaluation ===
    eval_dataset_name: str = "Salesforce/wikitext"
    eval_max_length: int = 10000
    eval_metric: Literal["perplexity", "loss"] = "perplexity"

    # === Device ===
    device: Optional[str] = None
    num_gpus: int = 1

    # === Reproducibility ===
    seed: int = 42

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Convert output_dir to Path
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate BPTT configuration
        if self.use_multi_stage:
            assert (
                self.stage1_bptt_segments > 0
            ), "stage1_bptt_segments must be positive"
            assert (
                self.stage2_bptt_segments > 0
            ), "stage2_bptt_segments must be positive"
            assert self.stage1_steps > 0, "stage1_steps must be positive"
            assert self.stage2_steps > 0, "stage2_steps must be positive"
        else:
            assert self.bptt_unroll_depth > 0, "bptt_unroll_depth must be positive"

        # Validate learning rate configuration
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.lr_decay_factor <= 1.0, "lr_decay_factor must be in (0, 1]"
        assert self.lr_decay_steps > 0, "lr_decay_steps must be positive"

        # Validate gradient clipping
        assert self.gradient_clip_norm > 0, "gradient_clip_norm must be positive"

        # Validate batch size
        assert self.batch_size > 0, "batch_size must be positive"

        # Auto-detect device if not specified
        if self.device is None:
            import torch

            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    def get_total_steps(self) -> int:
        """Get total training steps across all stages."""
        if self.use_multi_stage:
            return self.stage1_steps + self.stage2_steps
        else:
            # For single-stage, we use stage2_steps as the default
            return self.stage2_steps

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "backbone_model_name": self.backbone_model_name,
            "use_multi_stage": self.use_multi_stage,
            "stage1_steps": self.stage1_steps,
            "stage2_steps": self.stage2_steps,
            "stage1_bptt_segments": self.stage1_bptt_segments,
            "stage2_bptt_segments": self.stage2_bptt_segments,
            "bptt_unroll_depth": self.bptt_unroll_depth,
            "learning_rate": self.learning_rate,
            "lr_decay_factor": self.lr_decay_factor,
            "batch_size": self.batch_size,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_length": self.max_length,
            "device": self.device,
            "seed": self.seed,
        }


# === Preset Configurations (Table 7) ===


def get_opt_350m_config() -> TrainingConfig:
    """
    Training configuration for OPT 350M model.

    Paper Reference: Table 7, Row "OPT 350M / SMOLLM 135M"
    - Input Length: 2048
    - Segment Length: 1024
    - Learning Rate: 1E-5
    - LR Decay: 0.9
    """
    return TrainingConfig(
        backbone_model_name="facebook/opt-350m",
        learning_rate=1e-5,
        lr_decay_factor=0.9,
        max_length=2048,
        stage1_bptt_segments=2,  # 2048 / 1024 = 2
        stage2_bptt_segments=15,  # 15360 / 1024 = 15
    )


def get_opt_2_7b_config() -> TrainingConfig:
    """
    Training configuration for OPT 2.7B model.

    Paper Reference: Table 7, Row "OPT 2.7B / OPENLLAMAV2 3B"
    - Input Length: 2048
    - Segment Length: 512
    - Learning Rate: 1E-5
    - LR Decay: 0.9
    """
    return TrainingConfig(
        backbone_model_name="facebook/opt-2.7b",
        learning_rate=1e-5,
        lr_decay_factor=0.9,
        max_length=2048,
        stage1_bptt_segments=2,  # 1024 / 512 = 2
        stage2_bptt_segments=8,  # 4096 / 512 = 8
    )


def get_llama2_7b_config() -> TrainingConfig:
    """
    Training configuration for Llama 2 7B model.

    Paper Reference: Table 7, Row "LLAMA 2 7B"
    - Input Length: 2048
    - Segment Length: 256
    - Learning Rate: 1E-4
    - LR Decay: 0.7
    - Uses LoRA (rank 8) for parameter-efficient training
    """
    return TrainingConfig(
        backbone_model_name="meta-llama/Llama-2-7b-hf",
        learning_rate=1e-4,
        lr_decay_factor=0.7,
        max_length=2048,
        stage1_bptt_segments=2,  # 512 / 256 = 2
        stage2_bptt_segments=8,  # 2048 / 256 = 8
        use_lora=True,  # 7B model requires LoRA on limited VRAM
        gradient_checkpointing=True,
    )
