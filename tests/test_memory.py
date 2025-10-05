"""
Comprehensive tests for HMT memory components.

Tests cover:
1. RepresentationEncoder - segment summarization
2. MemorySearch - cross-attention retrieval
3. Edge cases and numerical stability
4. Device compatibility (MPS/CUDA/CPU)

Paper Reference: arXiv:2405.06067v3 [cs.CL] 6 Feb 2025
"""

import pytest
import torch
import torch.nn as nn

from hmt.memory import RepresentationEncoder, MemorySearch
from hmt.config import HMTConfig
from hmt.utils import get_device


@pytest.fixture(scope="module")
def device():
    """Get the appropriate device for testing."""
    return get_device()


@pytest.fixture
def config():
    """Create a standard HMT config for testing."""
    config = HMTConfig(
        segment_length=512,
        representation_length=256,  # j = L/2
        num_memory_embeddings=300,
        sensory_memory_size=32,
        hidden_dim=768,  # Standard transformer hidden size
    )
    return config


@pytest.fixture
def small_config():
    """Create a smaller config for faster tests."""
    config = HMTConfig(
        segment_length=64,
        representation_length=32,
        num_memory_embeddings=10,
        sensory_memory_size=8,
        hidden_dim=128,
    )
    return config


# ============================================================================
# RepresentationEncoder Tests
# ============================================================================

class TestRepresentationEncoder:
    """Test suite for RepresentationEncoder component."""

    def test_initialization(self, config):
        """Test that RepresentationEncoder initializes correctly."""
        encoder = RepresentationEncoder(config)

        assert encoder.hidden_dim == config.hidden_dim
        assert encoder.representation_length == config.representation_length
        assert hasattr(encoder, 'self_attention')
        assert hasattr(encoder, 'output_projection')

        print("\n  ✅ RepresentationEncoder initialized correctly")

    def test_output_shape(self, small_config, device):
        """Test that encoder produces correct output shape."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 4
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        # Create input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Encode
        summary = encoder(hidden_states)

        # Check shape
        assert summary.shape == (batch_size, hidden_dim), \
            f"Expected shape {(batch_size, hidden_dim)}, got {summary.shape}"

        print(f"\n  ✅ Output shape correct: {summary.shape}")

    def test_with_attention_mask(self, small_config, device):
        """Test encoder with attention mask (variable-length sequences)."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 4
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        # Create input with varying lengths
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create attention mask (1 = valid, 0 = padding)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        # Set different padding for each batch element
        attention_mask[0, 40:] = 0  # Pad after 40 tokens
        attention_mask[1, 50:] = 0  # Pad after 50 tokens
        attention_mask[2, :] = 1     # No padding
        attention_mask[3, 20:] = 0  # Pad after 20 tokens

        # Encode
        summary = encoder(hidden_states, attention_mask=attention_mask)

        # Check shape
        assert summary.shape == (batch_size, hidden_dim)

        # Check that outputs are different (masked vs unmasked)
        assert not torch.allclose(summary[0], summary[2]), \
            "Masked and unmasked outputs should differ"

        print(f"\n  ✅ Attention mask handling works correctly")

    def test_gradient_flow(self, small_config, device):
        """Test that gradients flow correctly through encoder."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        # Create input directly on device with requires_grad=True
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim,
            device=device,
            requires_grad=True
        )

        # Forward pass
        summary = encoder(hidden_states)

        # Create dummy loss
        loss = summary.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist (only for leaf tensors)
        assert hidden_states.grad is not None, "Input should have gradients"
        assert not torch.isnan(hidden_states.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(hidden_states.grad).any(), "Gradients should not be Inf"

        # Check that model parameters have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Gradients for {name} should not be NaN"

        print(f"\n  ✅ Gradient flow works correctly")

    def test_edge_case_single_token(self, small_config, device):
        """Test encoder with sequence of length 1."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 2
        seq_len = 1  # Edge case: single token
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Should still work
        summary = encoder(hidden_states)
        assert summary.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ Single token edge case handled")

    def test_edge_case_all_masked(self, small_config, device):
        """Test encoder when all tokens are masked."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create all-zero mask (all tokens masked)
        attention_mask = torch.zeros(batch_size, seq_len).to(device)

        # Should still work (avoid division by zero)
        summary = encoder(hidden_states, attention_mask=attention_mask)
        assert summary.shape == (batch_size, hidden_dim)
        assert not torch.isnan(summary).any(), "Output should not be NaN"

        print(f"\n  ✅ All-masked edge case handled")

    def test_variable_batch_sizes(self, small_config, device):
        """Test encoder with different batch sizes."""
        encoder = RepresentationEncoder(small_config).to(device)

        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        for batch_size in [1, 2, 4, 8]:
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
            summary = encoder(hidden_states)
            assert summary.shape == (batch_size, hidden_dim), \
                f"Failed for batch_size={batch_size}"

        print(f"\n  ✅ Variable batch sizes work correctly")

    def test_numerical_stability(self, small_config, device):
        """Test numerical stability with extreme values."""
        encoder = RepresentationEncoder(small_config).to(device)

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        # Test with large values
        hidden_states_large = torch.randn(batch_size, seq_len, hidden_dim).to(device) * 100
        summary_large = encoder(hidden_states_large)
        assert not torch.isnan(summary_large).any(), "Should handle large values"
        assert not torch.isinf(summary_large).any(), "Should handle large values"

        # Test with small values
        hidden_states_small = torch.randn(batch_size, seq_len, hidden_dim).to(device) * 0.01
        summary_small = encoder(hidden_states_small)
        assert not torch.isnan(summary_small).any(), "Should handle small values"

        print(f"\n  ✅ Numerical stability verified")

    def test_device_compatibility_cpu(self, small_config):
        """Test encoder on CPU."""
        encoder = RepresentationEncoder(small_config).to('cpu')

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        summary = encoder(hidden_states)

        assert summary.device.type == 'cpu'
        assert summary.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ CPU compatibility verified")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_compatibility_mps(self, small_config):
        """Test encoder on MPS (Apple Silicon)."""
        encoder = RepresentationEncoder(small_config).to('mps')

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to('mps')
        summary = encoder(hidden_states)

        assert summary.device.type == 'mps'
        assert summary.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ MPS compatibility verified")


# ============================================================================
# MemorySearch Tests
# ============================================================================

class TestMemorySearch:
    """Test suite for MemorySearch component."""

    def test_initialization(self, config):
        """Test that MemorySearch initializes correctly."""
        search = MemorySearch(config)

        assert search.hidden_dim == config.hidden_dim
        assert search.num_memory_embeddings == config.num_memory_embeddings
        assert hasattr(search, 'query_projection')
        assert hasattr(search, 'key_projection')

        print("\n  ✅ MemorySearch initialized correctly")

    def test_output_shape_with_cache(self, small_config, device):
        """Test memory search with non-empty cache."""
        search = MemorySearch(small_config).to(device)

        batch_size = 4
        cache_size = 10  # 10 past segments
        hidden_dim = small_config.hidden_dim

        # Create query (current segment representation)
        query = torch.randn(batch_size, hidden_dim).to(device)

        # Create memory cache
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to(device)

        # Retrieve
        retrieved = search(query, memory_cache)

        # Check shape
        assert retrieved.shape == (batch_size, hidden_dim), \
            f"Expected shape {(batch_size, hidden_dim)}, got {retrieved.shape}"

        print(f"\n  ✅ Output shape correct with cache: {retrieved.shape}")

    def test_empty_cache(self, small_config, device):
        """Test memory search with empty cache (first segment)."""
        search = MemorySearch(small_config).to(device)

        batch_size = 4
        hidden_dim = small_config.hidden_dim

        # Create query
        query = torch.randn(batch_size, hidden_dim).to(device)

        # No cache (None)
        retrieved = search(query, memory_cache=None)

        # Should return zero vector
        assert retrieved.shape == (batch_size, hidden_dim)
        assert torch.allclose(retrieved, torch.zeros_like(retrieved)), \
            "Empty cache should return zero vector"

        print(f"\n  ✅ Empty cache handled correctly")

    def test_cache_size_one(self, small_config, device):
        """Test with cache containing only one memory."""
        # Use eval mode to disable dropout
        search = MemorySearch(small_config).to(device)
        search.eval()

        batch_size = 2
        cache_size = 1  # Only one past segment
        hidden_dim = small_config.hidden_dim

        query = torch.randn(batch_size, hidden_dim).to(device)
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to(device)

        with torch.no_grad():
            retrieved = search(query, memory_cache)

        assert retrieved.shape == (batch_size, hidden_dim)
        # With cache_size=1, attention weight is 1.0 (softmax of single value)
        # Output should be close to cached memory (projections may introduce small differences)
        assert torch.allclose(retrieved, memory_cache.squeeze(1), rtol=0.1), \
            "With single cache item, output should be similar to cached memory"

        print(f"\n  ✅ Cache size = 1 edge case handled")

    def test_attention_weights_sum_to_one(self, small_config, device):
        """Test that attention weights sum to 1 (softmax property)."""
        search = MemorySearch(small_config).to(device)

        batch_size = 2
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        query = torch.randn(batch_size, hidden_dim).to(device)
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to(device)

        # We need to manually compute attention weights to verify
        with torch.no_grad():
            query_proj = search.query_projection(query).unsqueeze(1)
            keys_proj = search.key_projection(memory_cache)
            scores = torch.bmm(query_proj, keys_proj.transpose(1, 2)) * search.scale
            weights = torch.softmax(scores, dim=-1)

        # Check that weights sum to 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            "Attention weights should sum to 1.0"

        print(f"\n  ✅ Attention weights sum to 1.0")

    def test_gradient_flow(self, small_config, device):
        """Test gradient flow through memory search."""
        search = MemorySearch(small_config).to(device)

        batch_size = 2
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        # Create tensors directly on device as leaf tensors
        query = torch.randn(
            batch_size, hidden_dim,
            device=device,
            requires_grad=True
        )
        memory_cache = torch.randn(
            batch_size, cache_size, hidden_dim,
            device=device,
            requires_grad=True
        )

        # Forward
        retrieved = search(query, memory_cache)

        # Create loss
        loss = retrieved.sum()

        # Backward
        loss.backward()

        # Check gradients
        assert query.grad is not None, "Query should have gradients"
        assert memory_cache.grad is not None, "Memory cache should have gradients"
        assert not torch.isnan(query.grad).any(), "Query gradients should not be NaN"
        assert not torch.isnan(memory_cache.grad).any(), "Cache gradients should not be NaN"

        print(f"\n  ✅ Gradient flow works correctly")

    def test_cache_smaller_than_n(self, config, device):
        """Test when cache size is smaller than N (num_memory_embeddings)."""
        search = MemorySearch(config).to(device)

        batch_size = 2
        cache_size = 50  # Smaller than N=300
        hidden_dim = config.hidden_dim

        query = torch.randn(batch_size, hidden_dim).to(device)
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to(device)

        # Should work fine (use all available memories)
        retrieved = search(query, memory_cache)

        assert retrieved.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ Cache size < N handled correctly")

    def test_numerical_stability(self, small_config, device):
        """Test numerical stability with extreme values."""
        search = MemorySearch(small_config).to(device)

        batch_size = 2
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        # Test with large values
        query_large = torch.randn(batch_size, hidden_dim).to(device) * 100
        cache_large = torch.randn(batch_size, cache_size, hidden_dim).to(device) * 100

        retrieved_large = search(query_large, cache_large)
        assert not torch.isnan(retrieved_large).any(), "Should handle large values"
        assert not torch.isinf(retrieved_large).any(), "Should handle large values"

        # Test with small values
        query_small = torch.randn(batch_size, hidden_dim).to(device) * 0.01
        cache_small = torch.randn(batch_size, cache_size, hidden_dim).to(device) * 0.01

        retrieved_small = search(query_small, cache_small)
        assert not torch.isnan(retrieved_small).any(), "Should handle small values"

        print(f"\n  ✅ Numerical stability verified")

    def test_device_compatibility_cpu(self, small_config):
        """Test memory search on CPU."""
        search = MemorySearch(small_config).to('cpu')

        batch_size = 2
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        query = torch.randn(batch_size, hidden_dim)
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim)

        retrieved = search(query, memory_cache)

        assert retrieved.device.type == 'cpu'
        assert retrieved.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ CPU compatibility verified")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_compatibility_mps(self, small_config):
        """Test memory search on MPS (Apple Silicon)."""
        search = MemorySearch(small_config).to('mps')

        batch_size = 2
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        query = torch.randn(batch_size, hidden_dim).to('mps')
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to('mps')

        retrieved = search(query, memory_cache)

        assert retrieved.device.type == 'mps'
        assert retrieved.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ MPS compatibility verified")


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryComponentsIntegration:
    """Integration tests for RepresentationEncoder + MemorySearch."""

    def test_encoder_to_search_pipeline(self, small_config, device):
        """Test complete pipeline: encode segment → search memory."""
        encoder = RepresentationEncoder(small_config).to(device)
        search = MemorySearch(small_config).to(device)

        batch_size = 2
        seq_len = small_config.segment_length
        cache_size = 5
        hidden_dim = small_config.hidden_dim

        # Step 1: Encode current segment
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        segment_summary = encoder(hidden_states)

        # Step 2: Search memory cache
        memory_cache = torch.randn(batch_size, cache_size, hidden_dim).to(device)
        retrieved_memory = search(segment_summary, memory_cache)

        # Check shapes
        assert segment_summary.shape == (batch_size, hidden_dim)
        assert retrieved_memory.shape == (batch_size, hidden_dim)

        print(f"\n  ✅ Encoder → Search pipeline works correctly")

    def test_simulated_multi_segment_processing(self, small_config, device):
        """Simulate processing multiple segments with growing cache."""
        encoder = RepresentationEncoder(small_config).to(device)
        search = MemorySearch(small_config).to(device)

        batch_size = 2
        seq_len = small_config.segment_length
        hidden_dim = small_config.hidden_dim
        num_segments = 5

        # Simulate memory cache (list of embeddings)
        memory_cache_list = []

        for segment_idx in range(num_segments):
            # Create segment
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

            # Encode segment
            segment_summary = encoder(hidden_states)

            # Search memory (if cache exists)
            if len(memory_cache_list) > 0:
                # Stack cache
                memory_cache = torch.stack(memory_cache_list, dim=1)  # [batch, cache_size, hidden]
                retrieved = search(segment_summary, memory_cache)

                assert retrieved.shape == (batch_size, hidden_dim)
            else:
                # First segment, no cache
                retrieved = search(segment_summary, memory_cache=None)
                assert torch.allclose(retrieved, torch.zeros_like(retrieved))

            # Add to cache (simulate memory embedding generation)
            memory_cache_list.append(segment_summary)

        # Final cache should have num_segments items
        assert len(memory_cache_list) == num_segments

        print(f"\n  ✅ Multi-segment simulation works correctly")


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
