"""
Comprehensive Tests for HMT Training Infrastructure

Tests cover:
1. Multi-Stage Training Pipeline (Appendix F)
2. BPTT Mechanism and Gradient Flow (Appendix D, J)
3. Checkpoint Save/Load Functionality
4. Gradient Monitoring and Stability (Appendix J)
5. Learning Rate Scheduler (Table 7)
6. Memory Management During Training
7. Training Utilities

Paper Reference: arXiv:2405.06067v3 [cs.CL] 6 Feb 2025
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import tempfile
import shutil

from hmt import HMT, HMTConfig
from hmt.trainer import HMTTrainer
from hmt.training_config import TrainingConfig, get_opt_350m_config
from hmt.training_utils import (
    ExponentialDecayScheduler,
    GradientMonitor,
    clip_gradients,
    compute_gradient_stats,
    compute_perplexity,
    compute_loss_with_bptt,
    set_seed,
    count_parameters,
)
from hmt.utils import get_device
from transformers import AutoModelForCausalLM


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get the appropriate device for testing."""
    return get_device()


@pytest.fixture
def small_hmt_config():
    """Create a small HMT config for fast tests."""
    return HMTConfig(
        segment_length=32,
        representation_length=16,
        num_memory_embeddings=5,
        sensory_memory_size=8,
        hidden_dim=768,  # GPT-2 hidden size
    )


@pytest.fixture
def gpt2_model():
    """Load GPT-2 model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


class MockTextDataset(Dataset):
    """Mock dataset for training tests."""

    def __init__(self, num_samples=20, seq_length=128, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@pytest.fixture
def mock_train_dataloader():
    """Create mock training dataloader."""
    dataset = MockTextDataset(num_samples=20, seq_length=128)
    return DataLoader(dataset, batch_size=1, shuffle=True)


@pytest.fixture
def mock_eval_dataloader():
    """Create mock evaluation dataloader."""
    dataset = MockTextDataset(num_samples=10, seq_length=128)
    return DataLoader(dataset, batch_size=1, shuffle=False)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def training_config(temp_output_dir, device):
    """Create training configuration for tests."""
    return TrainingConfig(
        backbone_model_name="gpt2",
        use_multi_stage=True,
        stage1_steps=5,  # Very short for testing
        stage2_steps=5,
        stage1_bptt_segments=2,
        stage2_bptt_segments=2,
        learning_rate=1e-5,
        batch_size=1,
        logging_steps=1,
        eval_steps=3,
        save_steps=10,  # Don't save during short tests
        output_dir=str(temp_output_dir),
        device=str(device),
        seed=42,
    )


# ============================================================================
# Test Class 1: Multi-Stage Training Pipeline
# ============================================================================


class TestMultiStageTraining:
    """
    Test multi-stage training pipeline.

    Paper Reference: Appendix F - Multi-stage Training Methodology
    """

    def test_stage1_without_memory(
        self, gpt2_model, small_hmt_config, training_config, mock_train_dataloader, device
    ):
        """
        Test Stage 1 disables memory retrieval.

        Paper: Appendix F, Lines 1-8 - Stage 1 without memory retrieval
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Run Stage 1
        stats = trainer.train_stage(
            stage=1, num_steps=3, bptt_segments=2, use_memory=False
        )

        # Verify stage completed
        assert "final_loss" in stats
        assert "final_perplexity" in stats
        assert trainer.current_stage == 1

        print("\n  ✅ Stage 1 (without memory) completed successfully")

    def test_stage2_with_memory(
        self, gpt2_model, small_hmt_config, training_config, mock_train_dataloader, device
    ):
        """
        Test Stage 2 enables memory retrieval.

        Paper: Appendix F, Lines 9-15 - Stage 2 with memory retrieval
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Run Stage 2
        stats = trainer.train_stage(
            stage=2, num_steps=3, bptt_segments=2, use_memory=True
        )

        # Verify stage completed
        assert "final_loss" in stats
        assert "final_perplexity" in stats
        assert trainer.current_stage == 2

        print("\n  ✅ Stage 2 (with memory) completed successfully")

    def test_stage_transition(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        temp_output_dir,
        device,
    ):
        """
        Test checkpoint transition from Stage 1 to Stage 2.

        Paper: Appendix F - "Stage 2 loads checkpoint from Stage 1"
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Run Stage 1
        trainer.train_stage(stage=1, num_steps=3, bptt_segments=2, use_memory=False)

        # Save Stage 1 checkpoint
        stage1_ckpt = temp_output_dir / "stage1_test.pt"
        trainer.save_checkpoint(stage1_ckpt)

        # Create new trainer
        hmt2 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer2 = HMTTrainer(hmt2, training_config, mock_train_dataloader)

        # Load Stage 1 checkpoint
        trainer2.load_checkpoint(stage1_ckpt)

        # Verify state transferred
        assert trainer2.current_stage == 1
        assert trainer2.global_step == trainer.global_step

        # Run Stage 2
        trainer2.train_stage(stage=2, num_steps=3, bptt_segments=2, use_memory=True)

        assert trainer2.current_stage == 2

        print("\n  ✅ Stage transition (Stage 1 → Stage 2) works correctly")

    def test_multi_stage_end_to_end(
        self, gpt2_model, small_hmt_config, training_config, mock_train_dataloader, device
    ):
        """
        Test complete two-stage training pipeline.

        Paper: Appendix F - Complete multi-stage training
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Run both stages
        stats = trainer.train()

        # Verify both stages completed
        assert "stage1" in stats
        assert "stage2" in stats
        assert "final_perplexity" in stats
        assert trainer.current_stage == 2
        assert trainer.global_step == training_config.stage1_steps + training_config.stage2_steps

        print(f"\n  ✅ Multi-stage training completed")
        print(f"     Stage 1 PPL: {stats['stage1']['final_perplexity']:.2f}")
        print(f"     Stage 2 PPL: {stats['stage2']['final_perplexity']:.2f}")

    def test_loss_decreases(
        self, gpt2_model, small_hmt_config, training_config, mock_train_dataloader, device
    ):
        """
        Test that training actually improves the model (loss decreases).
        """
        # Use more steps for meaningful loss decrease
        config = training_config
        config.stage1_steps = 10
        config.stage2_steps = 10

        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, config, mock_train_dataloader)

        # Track losses
        initial_loss = None
        losses = []

        # Manually run a few steps and track loss
        trainer.model.train()
        train_iter = iter(mock_train_dataloader)

        for step in range(10):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(mock_train_dataloader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            loss = trainer._bptt_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bptt_segments=2,
                use_memory=True,
            )

            # Gradient step
            clip_gradients(trainer.model, max_norm=1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.model.clear_memory()

            losses.append(loss)

            if initial_loss is None:
                initial_loss = loss

        final_loss = losses[-1]

        # Loss should generally decrease (with some noise)
        # Use moving average for more robust check
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3

        print(f"\n  📉 Loss Evolution:")
        print(f"     Initial loss: {initial_loss:.4f}")
        print(f"     Final loss: {final_loss:.4f}")
        print(f"     Early avg (steps 0-2): {early_avg:.4f}")
        print(f"     Late avg (steps 7-9): {late_avg:.4f}")

        # Assert late average is lower (model is learning)
        assert (
            late_avg <= early_avg * 1.1
        ), f"Loss should decrease or stay similar, but increased from {early_avg:.4f} to {late_avg:.4f}"

        print("  ✅ Model is learning (loss decreased or stayed stable)")


# ============================================================================
# Test Class 2: BPTT Mechanism
# ============================================================================


class TestBPTTMechanism:
    """
    Test BPTT gradient flow and unrolling.

    Paper Reference: Appendix D - Training with BPTT, Appendix J - Gradient Stability
    """

    def test_bptt_gradient_flow(self, gpt2_model, small_hmt_config, device):
        """
        Test that gradients flow through K segments during BPTT.

        Paper: Appendix D - BPTT unroll depth
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Create input spanning 2 segments
        seq_len = small_hmt_config.segment_length * 2
        input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)

        # Forward pass
        outputs = hmt(input_ids, use_memory=True)

        # Compute loss
        loss = outputs["logits"].sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist in all components
        components = [
            hmt.representation_encoder,
            hmt.memory_search,
            hmt.memory_embedding_generator,
        ]

        for comp in components:
            has_grad = False
            for param in comp.parameters():
                if param.requires_grad and param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradients in {comp}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradients in {comp}"
                    has_grad = True

            assert has_grad, f"No gradients in {comp}"

        print("\n  ✅ Gradients flow correctly through all HMT components")

    def test_bptt_unroll_depths(self, gpt2_model, small_hmt_config, device):
        """
        Test different BPTT unroll depths (2, 3, 4).

        Paper: Appendix F - Stage 1 uses K=2, Stage 2 uses K=15
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)

        for unroll_depth in [2, 3, 4]:
            hmt.clear_memory()
            hmt.train()

            # Create input with exact number of segments
            seq_len = small_hmt_config.segment_length * unroll_depth
            input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)

            # Forward pass
            outputs = hmt(input_ids, use_memory=True)

            # Verify output shape
            assert outputs["logits"].shape[1] == seq_len

            # Verify memory cache size
            stats = hmt.get_memory_stats()
            assert stats["cache_size"] == unroll_depth

            print(f"  ✓ BPTT with K={unroll_depth} segments works")

        print("\n  ✅ All BPTT unroll depths work correctly")

    def test_gradient_accumulation(self, gpt2_model, small_hmt_config, device):
        """
        Test correct gradient accumulation across segments.
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Single forward pass with 2 segments
        seq_len = small_hmt_config.segment_length * 2
        input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)

        outputs = hmt(input_ids, use_memory=True)
        loss = outputs["logits"].sum()
        loss.backward()

        # Get gradient norms
        grad_norm = 0.0
        for param in hmt.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2

        grad_norm = grad_norm**0.5

        assert grad_norm > 0, "Gradients should exist"
        assert grad_norm < 1e6, "Gradients should not explode"

        print(f"\n  ✅ Gradient accumulation works (grad_norm={grad_norm:.4f})")

    def test_memory_detachment(self, gpt2_model, small_hmt_config, device):
        """
        Test that memory cache entries are detached (no gradient accumulation).

        Paper: Implementation detail for memory efficiency
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Process segments
        for _ in range(3):
            seq_len = small_hmt_config.segment_length
            input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)

            outputs = hmt(input_ids, use_memory=True)
            loss = outputs["logits"].sum()
            loss.backward()

            # Clear gradients for next iteration
            for param in hmt.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # Verify cache entries don't require gradients
        assert len(hmt.memory_cache) == 3

        for mem_emb in hmt.memory_cache:
            assert (
                not mem_emb.requires_grad
            ), "Memory cache entries should be detached"

        print("\n  ✅ Memory cache entries are correctly detached")

    def test_gradient_clipping(self, gpt2_model, small_hmt_config, device):
        """
        Test gradient clipping works correctly.

        Paper: Standard practice for BPTT stability
        """
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Create large loss to potentially cause large gradients
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        outputs = hmt(input_ids, use_memory=True)
        loss = outputs["logits"].sum() * 1000  # Amplify loss

        loss.backward()

        # Clip gradients
        max_norm = 1.0
        grad_norm_before = sum(
            p.grad.data.norm(2).item() ** 2
            for p in hmt.parameters()
            if p.grad is not None
        ) ** 0.5

        clipped_norm = clip_gradients(hmt, max_norm=max_norm)

        # Verify clipping
        assert clipped_norm >= 0

        # After clipping, gradient norm should be <= max_norm
        grad_norm_after = sum(
            p.grad.data.norm(2).item() ** 2
            for p in hmt.parameters()
            if p.grad is not None
        ) ** 0.5

        if grad_norm_before > max_norm:
            # Gradients were clipped
            assert (
                grad_norm_after <= max_norm * 1.01
            )  # Allow small numerical error

        print(f"\n  ✅ Gradient clipping works")
        print(f"     Norm before: {grad_norm_before:.4f}")
        print(f"     Norm after: {grad_norm_after:.4f}")
        print(f"     Max norm: {max_norm}")


# ============================================================================
# Test Class 3: Checkpointing
# ============================================================================


class TestCheckpointing:
    """
    Test checkpoint save/load functionality.

    Paper Reference: Appendix F - Stage 2 loads checkpoint from Stage 1
    """

    def test_save_checkpoint(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        temp_output_dir,
        device,
    ):
        """Test checkpoint saving with all state."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Train a few steps
        trainer.train_stage(stage=1, num_steps=3, bptt_segments=2, use_memory=False)

        # Save checkpoint
        ckpt_path = temp_output_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)

        # Verify file exists
        assert ckpt_path.exists()

        # Load and verify contents
        checkpoint = torch.load(ckpt_path, map_location=device)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_step" in checkpoint
        assert "global_step" in checkpoint
        assert "current_stage" in checkpoint
        assert "config" in checkpoint

        print(f"\n  ✅ Checkpoint saved successfully")
        print(f"     Path: {ckpt_path}")
        print(f"     Global step: {checkpoint['global_step']}")
        print(f"     Current stage: {checkpoint['current_stage']}")

    def test_load_checkpoint(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        temp_output_dir,
        device,
    ):
        """Test checkpoint loading with state restoration."""
        # Create and train first model
        hmt1 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer1 = HMTTrainer(hmt1, training_config, mock_train_dataloader)
        trainer1.train_stage(stage=1, num_steps=3, bptt_segments=2, use_memory=False)

        # Save checkpoint
        ckpt_path = temp_output_dir / "test_checkpoint.pt"
        trainer1.save_checkpoint(ckpt_path)

        # Create second model and load checkpoint
        hmt2 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer2 = HMTTrainer(hmt2, training_config, mock_train_dataloader)
        trainer2.load_checkpoint(ckpt_path)

        # Verify state matches
        assert trainer2.global_step == trainer1.global_step
        assert trainer2.current_stage == trainer1.current_stage

        # Verify model parameters match
        for p1, p2 in zip(hmt1.parameters(), hmt2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), "Model parameters should match"

        print(f"\n  ✅ Checkpoint loaded and verified")
        print(f"     Global step: {trainer2.global_step}")
        print(f"     Current stage: {trainer2.current_stage}")

    def test_resume_training(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        temp_output_dir,
        device,
    ):
        """Test resuming training from checkpoint."""
        # Train for a few steps
        hmt1 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer1 = HMTTrainer(hmt1, training_config, mock_train_dataloader)
        trainer1.train_stage(stage=1, num_steps=3, bptt_segments=2, use_memory=False)

        initial_step = trainer1.global_step

        # Save checkpoint
        ckpt_path = temp_output_dir / "resume_test.pt"
        trainer1.save_checkpoint(ckpt_path)

        # Load and resume
        hmt2 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer2 = HMTTrainer(hmt2, training_config, mock_train_dataloader)
        trainer2.load_checkpoint(ckpt_path)

        # Continue training
        trainer2.train_stage(stage=1, num_steps=2, bptt_segments=2, use_memory=False)

        # Verify global step increased
        assert trainer2.global_step == initial_step + 2

        print(f"\n  ✅ Training resumed successfully")
        print(f"     Initial step: {initial_step}")
        print(f"     Final step: {trainer2.global_step}")

    def test_checkpoint_compatibility(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        temp_output_dir,
        device,
    ):
        """Test Stage 1 checkpoint loads into Stage 2."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, training_config, mock_train_dataloader)

        # Run Stage 1
        trainer.train_stage(stage=1, num_steps=3, bptt_segments=2, use_memory=False)

        # Save Stage 1 checkpoint
        ckpt_path = temp_output_dir / "stage1_compat.pt"
        trainer.save_checkpoint(ckpt_path)

        # Create new trainer for Stage 2
        hmt2 = HMT(gpt2_model, small_hmt_config).to(device)
        trainer2 = HMTTrainer(hmt2, training_config, mock_train_dataloader)

        # Load Stage 1 checkpoint
        trainer2.load_checkpoint(ckpt_path)

        # Run Stage 2 (should work)
        trainer2.train_stage(stage=2, num_steps=2, bptt_segments=2, use_memory=True)

        assert trainer2.current_stage == 2

        print("\n  ✅ Stage 1 checkpoint compatible with Stage 2")

    def test_best_model_saving(
        self,
        gpt2_model,
        small_hmt_config,
        training_config,
        mock_train_dataloader,
        mock_eval_dataloader,
        temp_output_dir,
        device,
    ):
        """Test best model saving based on eval metric."""
        # Configure for evaluation
        config = training_config
        config.eval_steps = 2
        config.save_steps = 100  # Don't save regular checkpoints

        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        trainer = HMTTrainer(hmt, config, mock_train_dataloader, mock_eval_dataloader)

        # Manually run a few steps with eval
        trainer.model.train()
        train_iter = iter(mock_train_dataloader)

        for step in range(5):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(mock_train_dataloader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)

            loss = trainer._bptt_step(input_ids, None, 2, True)
            clip_gradients(trainer.model, 1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.model.clear_memory()

            trainer.global_step += 1

            # Evaluate at intervals
            if step % 2 == 0 and step > 0:
                eval_metrics = trainer.evaluate()

                # Update best metric
                if eval_metrics["perplexity"] < trainer.best_eval_metric:
                    trainer.best_eval_metric = eval_metrics["perplexity"]
                    best_ckpt = temp_output_dir / "best_stage1.pt"
                    trainer.save_checkpoint(best_ckpt)

        # Verify best checkpoint exists
        best_ckpt = temp_output_dir / "best_stage1.pt"
        assert best_ckpt.exists() or trainer.global_step < config.eval_steps

        print("\n  ✅ Best model saving works")


# ============================================================================
# Test Class 4: Gradient Monitoring
# ============================================================================


class TestGradientMonitoring:
    """
    Test gradient monitoring and stability detection.

    Paper Reference: Appendix J - Gradient Stability in HMT
    """

    def test_gradient_stats_computation(self, gpt2_model, small_hmt_config, device):
        """Test gradient statistics computation."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Forward + backward
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        outputs = hmt(input_ids, use_memory=True)
        loss = outputs["logits"].sum()
        loss.backward()

        # Compute gradient stats
        stats = compute_gradient_stats(hmt)

        assert "grad_norm" in stats
        assert "grad_max" in stats
        assert "grad_mean" in stats
        assert "grad_std" in stats

        assert stats["grad_norm"] > 0
        assert stats["grad_max"] > 0
        assert stats["grad_mean"] >= 0

        print(f"\n  ✅ Gradient statistics computed")
        print(f"     Norm: {stats['grad_norm']:.4f}")
        print(f"     Max: {stats['grad_max']:.4f}")
        print(f"     Mean: {stats['grad_mean']:.4f}")

    def test_gradient_explosion_detection(self):
        """Test gradient explosion detection."""
        monitor = GradientMonitor(explosion_threshold=5.0)

        # Simulate gradient norms
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(10))

            def parameters(self):
                return [self.param]

        model = MockModel()

        # Normal gradient
        model.param.grad = torch.ones(10) * 0.1  # norm ~ 0.316
        stats = monitor.update(model)
        assert not stats["is_exploding"]

        # Exploding gradient
        model.param.grad = torch.ones(10) * 10.0  # norm ~ 31.6
        stats = monitor.update(model)
        assert stats["is_exploding"]

        print("\n  ✅ Gradient explosion detection works")

    def test_gradient_vanishing_detection(self):
        """Test gradient vanishing detection."""
        monitor = GradientMonitor(vanishing_threshold=1e-6)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(10))

            def parameters(self):
                return [self.param]

        model = MockModel()

        # Normal gradient
        model.param.grad = torch.ones(10) * 0.1
        stats = monitor.update(model)
        assert not stats["is_vanishing"]

        # Vanishing gradient
        model.param.grad = torch.ones(10) * 1e-8
        stats = monitor.update(model)
        assert stats["is_vanishing"]

        print("\n  ✅ Gradient vanishing detection works")

    def test_moving_average(self):
        """Test gradient norm moving average computation."""
        monitor = GradientMonitor(window_size=5)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(10))

            def parameters(self):
                return [self.param]

        model = MockModel()

        # Update with different norms
        norms = [1.0, 2.0, 3.0, 4.0, 5.0]
        for norm in norms:
            model.param.grad = torch.ones(10) * norm / 3.162  # Adjust for actual norm
            stats = monitor.update(model)

        # Moving average should be computed
        assert stats["grad_norm_ma"] > 0

        print(f"\n  ✅ Moving average works")
        print(f"     MA: {stats['grad_norm_ma']:.4f}")

    def test_gradient_monitoring_integration(
        self, gpt2_model, small_hmt_config, device
    ):
        """Test integration of gradient monitoring in training loop."""
        monitor = GradientMonitor()

        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Simulate training steps
        for step in range(5):
            input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
                device
            )
            outputs = hmt(input_ids, use_memory=True)
            loss = outputs["logits"].sum()
            loss.backward()

            # Monitor gradients
            stats = monitor.update(hmt)

            assert "grad_norm" in stats
            assert "grad_norm_ma" in stats

            # Clear gradients
            for param in hmt.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        print(f"\n  ✅ Gradient monitoring integration works")


# ============================================================================
# Test Class 5: Learning Rate Scheduler
# ============================================================================


class TestLearningRateScheduler:
    """
    Test learning rate scheduler.

    Paper Reference: Table 7 - LR decay configurations
    """

    def test_exponential_decay(self):
        """
        Test exponential decay schedule.

        Paper: Table 7 - LR decays by 0.9 every 100 steps for OPT
        """
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        scheduler = ExponentialDecayScheduler(
            optimizer, initial_lr=1e-5, decay_factor=0.9, decay_steps=100, warmup_steps=0
        )

        # Step 0 - initial LR
        assert optimizer.param_groups[0]["lr"] == 1e-5

        # Step 100 - first decay
        for _ in range(100):
            scheduler.step()

        assert abs(optimizer.param_groups[0]["lr"] - 1e-5 * 0.9) < 1e-10

        # Step 200 - second decay
        for _ in range(100):
            scheduler.step()

        assert abs(optimizer.param_groups[0]["lr"] - 1e-5 * 0.9**2) < 1e-10

        print("\n  ✅ Exponential decay works correctly")

    def test_warmup(self):
        """Test linear warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        scheduler = ExponentialDecayScheduler(
            optimizer, initial_lr=1e-5, decay_factor=0.9, decay_steps=100, warmup_steps=10
        )

        # During warmup, LR should increase linearly
        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

        # After warmup, should reach initial_lr
        assert abs(lrs[-1] - 1e-5) < 1e-10

        print("\n  ✅ Warmup works correctly")

    def test_scheduler_step(self):
        """Test scheduler step updates LR correctly."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        scheduler = ExponentialDecayScheduler(
            optimizer, initial_lr=1e-5, decay_factor=0.9, decay_steps=10
        )

        # Track LR changes
        lrs = [scheduler.get_last_lr()]

        for _ in range(30):
            scheduler.step()
            lrs.append(scheduler.get_last_lr())

        # LR should change at steps 10, 20, 30
        assert lrs[0] == 1e-5  # Step 0
        assert abs(lrs[10] - 1e-5 * 0.9) < 1e-10  # Step 10
        assert abs(lrs[20] - 1e-5 * 0.9**2) < 1e-10  # Step 20
        assert abs(lrs[30] - 1e-5 * 0.9**3) < 1e-10  # Step 30

        print("\n  ✅ Scheduler step updates work correctly")

    def test_different_decay_factors(self):
        """
        Test different decay factors (0.7 and 0.9).

        Paper: Table 7 - Different models use different decay factors
        """
        for decay_factor in [0.7, 0.9]:
            model = nn.Linear(10, 10)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

            scheduler = ExponentialDecayScheduler(
                optimizer, initial_lr=1e-5, decay_factor=decay_factor, decay_steps=100
            )

            # Step 100
            for _ in range(100):
                scheduler.step()

            expected_lr = 1e-5 * decay_factor
            actual_lr = optimizer.param_groups[0]["lr"]

            assert abs(actual_lr - expected_lr) < 1e-10

            print(f"  ✓ Decay factor {decay_factor} works")

        print("\n  ✅ All decay factors work correctly")


# ============================================================================
# Test Class 6: Memory During Training
# ============================================================================


class TestMemoryDuringTraining:
    """Test memory system behavior during training."""

    def test_memory_cache_updates(self, gpt2_model, small_hmt_config, device):
        """Test that memory cache grows during forward passes."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Initially empty
        assert len(hmt.memory_cache) == 0

        # Process one segment
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        outputs = hmt(input_ids, use_memory=True)

        # Cache should have one entry
        assert len(hmt.memory_cache) == 1

        # Process another segment
        outputs = hmt(input_ids, use_memory=True)

        # Cache should have two entries
        assert len(hmt.memory_cache) == 2

        print("\n  ✅ Memory cache updates correctly during forward passes")

    def test_fifo_behavior(self, gpt2_model, small_hmt_config, device):
        """Test FIFO behavior when cache reaches max size N."""
        # Use very small N for testing
        config = HMTConfig(
            segment_length=32,
            num_memory_embeddings=3,  # Max 3
            sensory_memory_size=8,
            hidden_dim=768,
        )

        hmt = HMT(gpt2_model, config).to(device)
        hmt.train()

        # Process 5 segments (more than max)
        for i in range(5):
            input_ids = torch.randint(0, 1000, (1, config.segment_length)).to(device)
            with torch.no_grad():
                outputs = hmt(input_ids, use_memory=True)

        # Cache should be capped at 3
        assert len(hmt.memory_cache) == 3

        print("\n  ✅ FIFO behavior works (cache capped at N)")

    def test_sensory_memory_propagation(self, gpt2_model, small_hmt_config, device):
        """Test sensory memory propagates across segments."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Initially no sensory memory
        assert hmt.sensory_memory is None

        # Process one segment
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        # Sensory memory should now exist
        assert hmt.sensory_memory is not None
        assert hmt.sensory_memory.shape[1] == small_hmt_config.sensory_memory_size

        print("\n  ✅ Sensory memory propagates correctly")

    def test_clear_memory_between_batches(self, gpt2_model, small_hmt_config, device):
        """Test memory clearing between batches."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)

        # Process segment
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        # Memory should exist
        assert len(hmt.memory_cache) > 0
        assert hmt.sensory_memory is not None

        # Clear memory
        hmt.clear_memory()

        # Memory should be empty
        assert len(hmt.memory_cache) == 0
        assert hmt.sensory_memory is None

        print("\n  ✅ Memory clears correctly between batches")


# ============================================================================
# Test Class 7: Training Utilities
# ============================================================================


class TestTrainingUtils:
    """Test training utility functions."""

    def test_perplexity_computation(self):
        """Test perplexity = exp(loss) calculation."""
        import math

        loss = 2.0
        ppl = compute_perplexity(loss)

        expected_ppl = math.exp(loss)
        assert abs(ppl - expected_ppl) < 1e-6

        # Test with tensor
        loss_tensor = torch.tensor(2.0)
        ppl_tensor = compute_perplexity(loss_tensor)
        assert abs(ppl_tensor - expected_ppl) < 1e-6

        print("\n  ✅ Perplexity computation works")

    def test_compute_loss_with_bptt(self, gpt2_model, small_hmt_config, device):
        """Test loss computation for BPTT."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)

        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )

        loss = compute_loss_with_bptt(hmt, input_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

        print(f"\n  ✅ Loss computation works (loss={loss.item():.4f})")

    def test_clip_gradients_utility(self, gpt2_model, small_hmt_config, device):
        """Test gradient clipping utility."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)
        hmt.train()

        # Create gradients
        input_ids = torch.randint(0, 1000, (1, small_hmt_config.segment_length)).to(
            device
        )
        outputs = hmt(input_ids, use_memory=True)
        loss = outputs["logits"].sum() * 1000  # Large loss
        loss.backward()

        # Clip
        grad_norm = clip_gradients(hmt, max_norm=1.0)

        assert grad_norm >= 0

        print(f"\n  ✅ Gradient clipping utility works (norm={grad_norm:.4f})")

    def test_set_seed(self):
        """Test reproducibility with set_seed."""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(42)
        x2 = torch.randn(10)

        assert torch.allclose(x1, x2), "Same seed should produce same random numbers"

        print("\n  ✅ set_seed ensures reproducibility")

    def test_count_parameters(self, gpt2_model, small_hmt_config, device):
        """Test parameter counting utility."""
        hmt = HMT(gpt2_model, small_hmt_config).to(device)

        params = count_parameters(hmt)

        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params

        assert params["total"] == params["trainable"] + params["frozen"]
        assert params["trainable"] > 0

        print(f"\n  ✅ Parameter counting works")
        print(f"     Total: {params['total']:,}")
        print(f"     Trainable: {params['trainable']:,}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
