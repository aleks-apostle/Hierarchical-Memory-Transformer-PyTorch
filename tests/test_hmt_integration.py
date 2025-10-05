"""
Comprehensive integration tests for complete HMT system.

Tests cover:
1. MemoryEmbeddingGenerator - all extraction strategies
2. Full HMT forward pass - end-to-end processing
3. Memory cache management and FIFO behavior
4. Sensory memory propagation across segments
5. Three-level memory hierarchy integration
6. Gradient flow through complete pipeline
7. Ablation studies (with/without memory)
8. Edge cases and device compatibility

Paper Reference: arXiv:2405.06067v3 [cs.CL] 6 Feb 2025
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hmt import HMT, HMTConfig
from hmt.memory import MemoryEmbeddingGenerator
from hmt.utils import get_device


@pytest.fixture(scope="module")
def device():
    """Get the appropriate device for testing."""
    return get_device()


@pytest.fixture
def small_config():
    """Create a small config for faster tests."""
    config = HMTConfig(
        segment_length=64,
        representation_length=32,
        num_memory_embeddings=10,
        sensory_memory_size=8,
        hidden_dim=768,  # GPT-2 hidden size
    )
    return config


@pytest.fixture
def gpt2_model():
    """Load a small GPT-2 model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()  # Set to eval mode
    return model


@pytest.fixture
def tokenizer():
    """Load GPT-2 tokenizer."""
    return AutoTokenizer.from_pretrained("gpt2")


# ============================================================================
# MemoryEmbeddingGenerator Tests
# ============================================================================


class TestMemoryEmbeddingGenerator:
    """Test suite for MemoryEmbeddingGenerator component."""

    def test_initialization(self, small_config):
        """Test that MemoryEmbeddingGenerator initializes correctly."""
        generator = MemoryEmbeddingGenerator(small_config)

        assert generator.hidden_dim == small_config.hidden_dim
        assert hasattr(generator, "compression")

        print("\n  ✅ MemoryEmbeddingGenerator initialized correctly")

    def test_extraction_strategy_last(self, small_config, device):
        """Test 'last' token extraction strategy."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 4
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        # Create backbone hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Extract memory embedding
        memory_emb = generator(hidden_states, extraction_strategy="last")

        # Check shape
        assert memory_emb.shape == (
            batch_size,
            hidden_dim,
        ), f"Expected shape {(batch_size, hidden_dim)}, got {memory_emb.shape}"

        print(f"\n  ✅ 'last' extraction strategy works: {memory_emb.shape}")

    def test_extraction_strategy_mean(self, small_config, device):
        """Test 'mean' pooling extraction strategy."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 4
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        memory_emb = generator(hidden_states, extraction_strategy="mean")

        assert memory_emb.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ 'mean' extraction strategy works: {memory_emb.shape}")

    def test_extraction_strategy_max(self, small_config, device):
        """Test 'max' pooling extraction strategy."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 4
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        memory_emb = generator(hidden_states, extraction_strategy="max")

        assert memory_emb.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ 'max' extraction strategy works: {memory_emb.shape}")

    def test_extraction_strategy_cls(self, small_config, device):
        """Test 'cls' (first token) extraction strategy."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 4
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        memory_emb = generator(hidden_states, extraction_strategy="cls")

        assert memory_emb.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ 'cls' extraction strategy works: {memory_emb.shape}")

    def test_with_attention_mask(self, small_config, device):
        """Test extraction with attention mask (variable-length sequences)."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 4
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create attention mask with different padding
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        attention_mask[0, 30:] = 0  # Pad after 30 tokens
        attention_mask[1, 40:] = 0  # Pad after 40 tokens
        attention_mask[2, :] = 1  # No padding
        attention_mask[3, 20:] = 0  # Pad after 20 tokens

        # Test all strategies with mask
        for strategy in ["last", "mean", "max"]:
            memory_emb = generator(
                hidden_states, extraction_strategy=strategy, attention_mask=attention_mask
            )
            assert memory_emb.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ Attention mask handling works for all strategies")

    def test_gradient_flow(self, small_config, device):
        """Test gradient flow through memory embedding generator."""
        generator = MemoryEmbeddingGenerator(small_config).to(device)

        batch_size = 2
        seq_len = 50
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, requires_grad=True
        )

        # Forward
        memory_emb = generator(hidden_states, extraction_strategy="last")

        # Create loss
        loss = memory_emb.sum()

        # Backward
        loss.backward()

        # Check gradients
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

        # Check model parameters
        for name, param in generator.named_parameters():
            assert param.grad is not None, f"Parameter {name} missing gradients"

        print(f"\n  ✅ Gradient flow works correctly")


# ============================================================================
# Full HMT Integration Tests
# ============================================================================


class TestHMTIntegration:
    """Test suite for complete HMT system."""

    def test_hmt_initialization(self, gpt2_model, small_config):
        """Test HMT wrapper initialization."""
        hmt = HMT(gpt2_model, small_config)

        assert hmt.config.hidden_dim == 768  # GPT-2 hidden size
        assert hasattr(hmt, "representation_encoder")
        assert hasattr(hmt, "memory_search")
        assert hasattr(hmt, "memory_embedding_generator")
        assert hasattr(hmt, "memory_cache")
        assert hasattr(hmt, "sensory_memory")

        print("\n  ✅ HMT initialized correctly with all components")

    def test_single_segment_forward(self, gpt2_model, small_config, device):
        """Test forward pass with single segment (seq_len < segment_length)."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 2
        seq_len = 50  # Less than segment_length (64)

        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = hmt(input_ids, return_dict=True, use_memory=True)

        # Check outputs
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 50257)  # GPT-2 vocab size

        # Check memory state
        stats = hmt.get_memory_stats()
        assert stats["cache_size"] == 1  # One segment processed
        assert stats["sensory_memory_active"] == True

        print(f"\n  ✅ Single segment forward pass works")
        print(f"     Memory stats: {stats}")

    def test_multi_segment_forward(self, gpt2_model, small_config, device):
        """Test forward pass with multiple segments."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 2
        seq_len = 200  # 200 tokens = 4 segments (64 * 3 + 8)

        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = hmt(input_ids, return_dict=True, use_memory=True)

        # Check outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 50257)

        # Check memory cache growth
        stats = hmt.get_memory_stats()
        expected_segments = (
            seq_len + small_config.segment_length - 1
        ) // small_config.segment_length
        assert stats["cache_size"] == expected_segments

        print(f"\n  ✅ Multi-segment forward pass works")
        print(f"     Processed {expected_segments} segments")
        print(f"     Memory stats: {stats}")

    def test_memory_cache_fifo(self, gpt2_model, small_config, device):
        """Test that memory cache follows FIFO with max size N."""
        # Use very small cache for testing
        config = HMTConfig(
            segment_length=32,
            num_memory_embeddings=3,  # Max 3 embeddings
            sensory_memory_size=8,
            hidden_dim=768,
        )

        hmt = HMT(gpt2_model, config).to(device)
        hmt.eval()

        batch_size = 1
        seq_len = 32 * 5  # 5 segments

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            outputs = hmt(input_ids, return_dict=True, use_memory=True)

        # Cache should be capped at max size (3)
        stats = hmt.get_memory_stats()
        assert (
            stats["cache_size"] == 3
        ), f"Cache should be capped at {config.num_memory_embeddings}, got {stats['cache_size']}"

        print(f"\n  ✅ Memory cache FIFO behavior works (max size={config.num_memory_embeddings})")

    def test_sensory_memory_propagation(self, gpt2_model, small_config, device):
        """Test sensory memory propagates across segments."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 1
        seq_len = 150  # Multiple segments

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Clear memory first
        hmt.clear_memory()
        assert hmt.sensory_memory is None

        with torch.no_grad():
            outputs = hmt(input_ids, return_dict=True, use_memory=True)

        # After processing, sensory memory should exist
        assert hmt.sensory_memory is not None
        assert hmt.sensory_memory.shape[1] == small_config.sensory_memory_size

        print(f"\n  ✅ Sensory memory propagation works")
        print(f"     Sensory memory shape: {hmt.sensory_memory.shape}")

    def test_ablation_with_without_memory(self, gpt2_model, small_config, device):
        """Test ablation: compare outputs with and without memory."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 2
        seq_len = 150

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Forward with memory
        hmt.clear_memory()
        with torch.no_grad():
            outputs_with_memory = hmt(input_ids, use_memory=True)

        # Forward without memory
        hmt.clear_memory()
        with torch.no_grad():
            outputs_without_memory = hmt(input_ids, use_memory=False)

        # Both should produce valid outputs
        assert outputs_with_memory["logits"].shape == outputs_without_memory["logits"].shape

        # Outputs should be different (memory affects processing)
        assert not torch.allclose(
            outputs_with_memory["logits"], outputs_without_memory["logits"], atol=1e-4
        ), "Outputs should differ when memory is enabled vs disabled"

        print(f"\n  ✅ Ablation study works (with vs without memory)")

    def test_clear_memory(self, gpt2_model, small_config, device):
        """Test memory clearing between sequences."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 1
        seq_len = 100

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Process sequence
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

        print(f"\n  ✅ Memory clearing works correctly")

    def test_with_actual_text(self, gpt2_model, tokenizer, small_config, device):
        """Test HMT with actual text input."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        # Sample text
        text = """
        The Hierarchical Memory Transformer is a novel architecture designed to handle
        long-context language processing efficiently. It uses a three-level memory
        hierarchy inspired by human cognition: sensory memory for immediate context,
        short-term memory for current processing, and long-term memory for distant
        information. This approach reduces computational complexity while maintaining
        the ability to access relevant information from the entire context.
        """

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        # Check outputs
        assert "logits" in outputs
        assert outputs["logits"].shape[0] == 1
        assert outputs["logits"].shape[1] == input_ids.shape[1]

        # Check memory was used
        stats = hmt.get_memory_stats()
        assert stats["cache_size"] > 0

        print(f"\n  ✅ HMT works with actual text")
        print(f"     Input length: {input_ids.shape[1]} tokens")
        print(f"     Memory stats: {stats}")

    def test_gradient_flow_full_pipeline(self, gpt2_model, small_config, device):
        """Test gradient flow through complete HMT pipeline."""
        # Use smaller model for gradient test
        config = HMTConfig(
            segment_length=32,
            num_memory_embeddings=5,
            sensory_memory_size=8,
            hidden_dim=768,
        )

        hmt = HMT(gpt2_model, config).to(device)
        hmt.train()  # Set to training mode

        batch_size = 2
        seq_len = 100

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # Forward pass
        outputs = hmt(input_ids, use_memory=True)

        # Create dummy loss
        loss = outputs["logits"].sum()

        # Backward pass
        loss.backward()

        # Check that model parameters have gradients
        has_gradients = False
        for name, param in hmt.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"
                has_gradients = True

        assert has_gradients, "No gradients computed in the model"

        print(f"\n  ✅ Gradient flow works through complete HMT pipeline")

    def test_edge_case_very_short_input(self, gpt2_model, small_config, device):
        """Test with very short input (< sensory_memory_size)."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 1
        seq_len = 5  # Very short

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        assert outputs["logits"].shape == (batch_size, seq_len, 50257)

        print(f"\n  ✅ Edge case: very short input handled")

    def test_edge_case_exact_segment_length(self, gpt2_model, small_config, device):
        """Test with input exactly equal to segment_length."""
        hmt = HMT(gpt2_model, small_config).to(device)
        hmt.eval()

        batch_size = 2
        seq_len = small_config.segment_length  # Exactly one segment

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        assert outputs["logits"].shape == (batch_size, seq_len, 50257)

        stats = hmt.get_memory_stats()
        assert stats["cache_size"] == 1

        print(f"\n  ✅ Edge case: exact segment length handled")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_compatibility_mps(self, gpt2_model, small_config):
        """Test HMT on MPS (Apple Silicon)."""
        hmt = HMT(gpt2_model, small_config).to("mps")
        hmt.eval()

        batch_size = 2
        seq_len = 100

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to("mps")

        with torch.no_grad():
            outputs = hmt(input_ids, use_memory=True)

        assert outputs["logits"].device.type == "mps"

        print(f"\n  ✅ MPS compatibility verified")


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
