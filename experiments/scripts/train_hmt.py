#!/usr/bin/env python3
"""
HMT Training Script

Train an HMT model with multi-stage BPTT on WikiText-103 or custom datasets.

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

Usage Examples:
    # Train HMT with OPT-350M backbone (default configuration from Table 7)
    python experiments/scripts/train_hmt.py --model facebook/opt-350m

    # Train with custom configuration
    python experiments/scripts/train_hmt.py \\
        --model facebook/opt-2.7b \\
        --stage1-steps 300 \\
        --stage2-steps 700 \\
        --learning-rate 1e-4 \\
        --output-dir ./checkpoints/opt2.7b

    # Resume from checkpoint
    python experiments/scripts/train_hmt.py \\
        --model facebook/opt-350m \\
        --resume ./checkpoints/checkpoint_step500.pt

    # Single-stage training (no multi-stage)
    python experiments/scripts/train_hmt.py \\
        --model facebook/opt-350m \\
        --no-multi-stage \\
        --bptt-unroll 10

    # Enable W&B logging
    python experiments/scripts/train_hmt.py \\
        --model facebook/opt-350m \\
        --use-wandb \\
        --wandb-project my-hmt-project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from hmt import (
    HMT,
    HMTConfig,
    HMTTrainer,
    TrainingConfig,
    WikiTextDataset,
    LongContextDataLoader,
    get_device,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HMT with multi-stage BPTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # === Model Configuration ===
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="HuggingFace model name or path (default: facebook/opt-350m)",
    )
    model_group.add_argument(
        "--segment-length",
        type=int,
        default=512,
        help="Segment length L (default: 512, from Table 7)",
    )
    model_group.add_argument(
        "--num-memory-embeddings",
        type=int,
        default=300,
        help="Number of cached memory embeddings N (default: 300)",
    )
    model_group.add_argument(
        "--sensory-memory-size",
        type=int,
        default=32,
        help="Sensory memory size k (default: 32)",
    )

    # === Training Configuration ===
    train_group = parser.add_argument_group("Training Configuration (Appendix D, F)")
    train_group.add_argument(
        "--no-multi-stage",
        action="store_true",
        help="Disable multi-stage training (use single-stage BPTT)",
    )
    train_group.add_argument(
        "--stage1-steps",
        type=int,
        default=200,
        help="Stage 1 training steps (default: 200, Appendix F)",
    )
    train_group.add_argument(
        "--stage2-steps",
        type=int,
        default=500,
        help="Stage 2 training steps (default: 500, Appendix F)",
    )
    train_group.add_argument(
        "--stage1-bptt-segments",
        type=int,
        default=2,
        help="BPTT unroll depth for Stage 1 (default: 2)",
    )
    train_group.add_argument(
        "--stage2-bptt-segments",
        type=int,
        default=15,
        help="BPTT unroll depth for Stage 2 (default: 15)",
    )
    train_group.add_argument(
        "--bptt-unroll",
        type=int,
        default=4,
        help="BPTT unroll depth for single-stage training (default: 4)",
    )

    # === Optimization ===
    opt_group = parser.add_argument_group("Optimization (Table 7)")
    opt_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, Table 7)",
    )
    opt_group.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.9,
        help="LR decay factor (default: 0.9 for OPT, 0.7 for others)",
    )
    opt_group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2, Table 7)",
    )
    opt_group.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: 1.0)",
    )

    # === Data ===
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="Salesforce/wikitext",
        choices=["Salesforce/wikitext", "togethercomputer/RedPajama-Data-V2"],
        help="Dataset to use (default: Salesforce/wikitext)",
    )
    data_group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # === Output and Logging ===
    output_group = parser.add_argument_group("Output and Logging")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints (default: ./checkpoints)",
    )
    output_group.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    output_group.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps (default: 100)",
    )
    output_group.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )

    # === W&B ===
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="hmt-training",
        help="W&B project name (default: hmt-training)",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )

    # === Checkpointing ===
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )

    # === Other ===
    other_group = parser.add_argument_group("Other")
    other_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    other_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detect if None)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("HMT Training Script")
    print("Paper: arXiv:2405.06067v3")
    print("=" * 80)

    # === Load Backbone Model ===
    print(f"\nLoading backbone model: {args.model}")
    backbone_model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Backbone loaded: {args.model}")
    print(f"  Hidden size: {backbone_model.config.hidden_size}")
    print(f"  Vocab size: {backbone_model.config.vocab_size}")

    # === Create HMT Model ===
    print("\nCreating HMT model...")
    hmt_config = HMTConfig(
        segment_length=args.segment_length,
        num_memory_embeddings=args.num_memory_embeddings,
        sensory_memory_size=args.sensory_memory_size,
        hidden_dim=backbone_model.config.hidden_size,
    )
    hmt_model = HMT(backbone_model, hmt_config)
    print(f"✓ HMT model created")
    print(f"  Segment length (L): {hmt_config.segment_length}")
    print(f"  Memory cache size (N): {hmt_config.num_memory_embeddings}")
    print(f"  Sensory memory (k): {hmt_config.sensory_memory_size}")

    # === Load Data ===
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "Salesforce/wikitext":
        train_dataset = WikiTextDataset(
            split="train",
            tokenizer=tokenizer,
            max_length=args.max_length,
            min_length=args.segment_length,
        )
        eval_dataset = WikiTextDataset(
            split="validation",
            tokenizer=tokenizer,
            max_length=10000,  # Longer for eval
            min_length=args.segment_length,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported")

    train_loader = LongContextDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    eval_loader = LongContextDataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    print(f"✓ Dataset loaded")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")

    # === Create Training Config ===
    training_config = TrainingConfig(
        backbone_model_name=args.model,
        # Multi-stage
        use_multi_stage=not args.no_multi_stage,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        stage1_bptt_segments=args.stage1_bptt_segments,
        stage2_bptt_segments=args.stage2_bptt_segments,
        bptt_unroll_depth=args.bptt_unroll,
        # Optimization
        learning_rate=args.learning_rate,
        lr_decay_factor=args.lr_decay_factor,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        # Data
        dataset_name=args.dataset,
        max_length=args.max_length,
        # Output
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        # W&B
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        # Other
        seed=args.seed,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for key, value in training_config.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # === Create Trainer ===
    trainer = HMTTrainer(
        model=hmt_model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    # === Resume from Checkpoint (if specified) ===
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # === Train ===
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    stats = trainer.train()

    # === Final Summary ===
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    if training_config.use_multi_stage:
        print(f"Stage 1 Final PPL: {stats['stage1']['final_perplexity']:.2f}")
        print(f"Stage 2 Final PPL: {stats['stage2']['final_perplexity']:.2f}")
    print(f"Final Perplexity: {stats['final_perplexity']:.2f}")
    print(f"Best Eval Metric: {stats['best_eval_metric']:.4f}")
    print(f"Checkpoints saved to: {training_config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
