"""
HMT Memory Components

This module contains the three core memory components for HMT:
- RepresentationEncoder: Encodes segment into summary embedding (Section 3.2, Eq. 1)
- MemorySearch: Cross-attention based memory retrieval (Section 3.2, Eq. 2-3)
- MemoryEmbeddingGenerator: Generates memory embeddings from segments (Section 3.3)

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class RepresentationEncoder(nn.Module):
    """
    Representation Encoder for HMT Memory System.

    Paper Reference:
      - Section 3.2: "Memory Retrieval Mechanism" → "Representation Encoding"
      - Equation 1: S_n = BBM([T||H_n[0, j)||T])[j, j + 1)
      - Figure 1: Step 1 - Representation encoding

    Encodes the first j tokens (j = L/2) of a segment into a fixed-size summary
    embedding for memory search. The paper uses the backbone model directly,
    but this implementation uses multi-head self-attention + mean pooling for
    efficiency and model-independence.

    Args:
        config (HMTConfig): Configuration with segment_length, representation_length,
                           and hidden_dim

    Implementation Notes:
      - Paper specifies: Uses backbone model to process [T||H_n[0,j)||T]
      - Our approach: Self-attention + pooling (more efficient, model-independent)
      - Rationale: Allows parallel execution and doesn't require full backbone pass
      - Both approaches achieve the same goal: compress first j tokens into summary
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.representation_length = config.representation_length  # j in paper
        self.hidden_dim = config.hidden_dim

        # Multi-head self-attention for encoding first j tokens
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,  # Standard choice, can be tuned
            dropout=0.1,
            batch_first=True,  # Input shape: [batch, seq, hidden_dim]
        )

        # Layer normalization for training stability
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        # Feed-forward network for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(0.1),
        )

        # Final projection to ensure correct output dimension
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Initialize weights with Xavier uniform for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode segment representation from first j tokens.

        Args:
            hidden_states: Hidden embeddings of segment
                          Shape: [batch_size, seq_len, hidden_dim]
            attention_mask: Mask for padding tokens (1 = valid, 0 = padding)
                           Shape: [batch_size, seq_len]

        Returns:
            Segment summary embedding
            Shape: [batch_size, hidden_dim]

        Implementation:
            1. Extract first j tokens from segment
            2. Apply self-attention to capture token relationships
            3. Mean pool over sequence dimension (masked)
            4. Project to output space
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Step 1: Extract first j tokens (Paper Eq. 1: H_n[0, j))
        j = min(self.representation_length, seq_len)
        first_j_tokens = hidden_states[:, :j, :]  # [batch, j, hidden_dim]

        # Handle attention mask for first j tokens
        if attention_mask is not None:
            first_j_mask = attention_mask[:, :j]  # [batch, j]
            # Convert to attention mask format for nn.MultiheadAttention
            # True = masked (ignore), False = attend
            attn_mask_for_mha = ~first_j_mask.bool()  # Invert: 0 -> True, 1 -> False
        else:
            attn_mask_for_mha = None
            first_j_mask = torch.ones(batch_size, j, dtype=torch.bool, device=hidden_states.device)

        # Step 2: Self-attention over first j tokens
        # This captures relationships between tokens for better summarization
        attn_output, _ = self.self_attention(
            query=first_j_tokens,
            key=first_j_tokens,
            value=first_j_tokens,
            key_padding_mask=attn_mask_for_mha if attn_mask_for_mha is not None else None,
            need_weights=False,
        )

        # Residual connection + layer norm
        first_j_tokens = self.layer_norm1(first_j_tokens + attn_output)

        # Feed-forward network with residual
        ffn_output = self.ffn(first_j_tokens)
        first_j_tokens = self.layer_norm2(first_j_tokens + ffn_output)

        # Step 3: Mean pooling over sequence dimension (masked)
        # This compresses the j tokens into a single summary embedding
        if attention_mask is not None:
            # Masked mean: sum(hidden * mask) / sum(mask)
            mask_expanded = first_j_mask.unsqueeze(-1).float()  # [batch, j, 1]
            masked_hidden = first_j_tokens * mask_expanded
            sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = mask_expanded.sum(dim=1)  # [batch, 1]
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            summary = sum_hidden / sum_mask  # [batch, hidden_dim]
        else:
            # Simple mean pooling
            summary = first_j_tokens.mean(dim=1)  # [batch, hidden_dim]

        # Step 4: Final projection (Paper: generates S_n)
        summary = self.output_projection(summary)  # [batch, hidden_dim]

        return summary


class MemorySearch(nn.Module):
    """
    Memory Search via Cross-Attention for HMT.

    Paper Reference:
      - Section 3.2: "Memory Retrieval Mechanism" → "Memory Search"
      - Equation 2: Q_n = S_n W_q, K_n = M[n-N+1,n) W_k
      - Equation 3: P_n = softmax(Q_n K_n^T / sqrt(d_h)) M[n-N+1,n)
      - Figure 1: Step 2 - Memory search with cross attention

    Retrieves relevant memories from long-term cache using cross-attention.
    Query is current segment representation (S_n from RepresentationEncoder),
    keys/values are cached memory embeddings from previous segments.

    Args:
        config (HMTConfig): Configuration with hidden_dim and num_memory_embeddings

    Implementation Notes:
      - Paper specifies: Cross-attention WITHOUT value projection and output projection
      - Uses scaled dot-product attention with softmax normalization
      - Directly applies attention weights to memory cache
      - This ensures similar distributions between output and cached memories
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_memory_embeddings = config.num_memory_embeddings  # N in paper

        # Query and Key projections (Paper Eq. 2: W_q and W_k)
        # Note: No value projection per paper specification
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Scaling factor for dot-product attention (Paper Eq. 3: 1/sqrt(d_h))
        self.scale = 1.0 / math.sqrt(self.hidden_dim)

        # Dropout for attention weights (regularization)
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection matrices with Xavier uniform."""
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(
        self,
        query: torch.Tensor,
        memory_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retrieve relevant memories using cross-attention.

        Args:
            query: Current segment representation (S_n from RepresentationEncoder)
                  Shape: [batch_size, hidden_dim]
            memory_cache: Cached memory embeddings from previous segments
                         Shape: [batch_size, cache_size, hidden_dim]
                         where cache_size <= N (num_memory_embeddings)
                         Can be None for first segment (no history yet)

        Returns:
            Retrieved memory embedding (P_n in paper)
            Shape: [batch_size, hidden_dim]

        Implementation (Paper Eq. 2-3):
            1. Project query and keys: Q_n = S_n W_q, K_n = M W_k
            2. Compute attention scores: scores = Q_n @ K_n^T / sqrt(d_h)
            3. Apply softmax for normalization
            4. Weighted sum of memory cache: P_n = softmax(scores) @ M

        Edge Cases:
            - Empty cache (first segment): Returns zero vector
            - Cache size < N: Uses all available memories
            - Cache size > N: Should be managed by caller (sliding window)
        """
        batch_size = query.shape[0]

        # Edge case: No memory cache yet (first segment, t=0)
        if memory_cache is None or memory_cache.size(1) == 0:
            # Return zero embedding (no past context to retrieve)
            return torch.zeros(batch_size, self.hidden_dim, dtype=query.dtype, device=query.device)

        cache_size = memory_cache.size(1)  # Number of cached memories

        # Step 1: Project query and keys (Paper Eq. 2)
        # Q_n = S_n W_q
        query_projected = self.query_projection(query)  # [batch, hidden_dim]
        query_projected = query_projected.unsqueeze(1)  # [batch, 1, hidden_dim]

        # K_n = M[n-N+1,n) W_k
        keys_projected = self.key_projection(memory_cache)  # [batch, cache_size, hidden_dim]

        # Step 2: Compute attention scores (Paper Eq. 3: Q_n K_n^T / sqrt(d_h))
        # Scaled dot-product attention
        attention_scores = torch.bmm(
            query_projected,  # [batch, 1, hidden_dim]
            keys_projected.transpose(1, 2),  # [batch, hidden_dim, cache_size]
        )  # -> [batch, 1, cache_size]

        attention_scores = attention_scores * self.scale  # Scale by 1/sqrt(d_h)

        # Step 3: Softmax normalization (Paper Eq. 3: softmax(...))
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch, 1, cache_size]
        attention_weights = self.dropout(attention_weights)

        # Step 4: Weighted sum of memory cache (Paper Eq. 3: ... @ M[n-N+1,n))
        # P_n = attention_weights @ M
        # Note: Paper directly applies weights to M (no value projection)
        retrieved_memory = torch.bmm(
            attention_weights,  # [batch, 1, cache_size]
            memory_cache,  # [batch, cache_size, hidden_dim]
        )  # -> [batch, 1, hidden_dim]

        retrieved_memory = retrieved_memory.squeeze(1)  # [batch, hidden_dim]

        return retrieved_memory


class MemoryEmbeddingGenerator(nn.Module):
    """
    Memory Embedding Generator for HMT.

    Paper Reference:
      - Section 3.3: "Memory Embedding Generation"
      - Equation 4: m_n = compress(BBM([k_n||H_n||P_n]))
      - Figure 1: Step 3 - Memory embedding generation

    Generates compressed memory embeddings from augmented segments to store in
    the long-term memory cache. The augmented segment consists of:
      - k_n: Sensory memory (last k tokens from previous segment)
      - H_n: Current segment (L tokens)
      - P_n: Retrieved memory embedding from MemorySearch

    The backbone model processes this augmented input, and this module extracts
    a compressed representation (memory embedding) to cache for future retrieval.

    Args:
        config (HMTConfig): Configuration with hidden_dim and compression settings

    Implementation Notes:
      - Paper approach: Use backbone's output at specific position (often last token)
      - Additional compression: Optional learnable projection for dimensionality control
      - Cache storage: Output becomes part of M[n] for future segments
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Compression strategy: learnable projection
        # This allows the model to learn what information to preserve in memory
        self.compression = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Alternative: Identity mapping (no compression, just use backbone output)
        # Uncomment to disable learned compression:
        # self.compression = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        backbone_hidden_states: torch.Tensor,
        extraction_strategy: str = "last",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate memory embedding from backbone's hidden states.

        Args:
            backbone_hidden_states: Hidden states from backbone model processing
                                   augmented segment [k_n||H_n||P_n]
                                   Shape: [batch_size, seq_len, hidden_dim]
            extraction_strategy: How to extract memory from hidden states
                                "last" - Use last token (default, paper approach)
                                "mean" - Mean pool over sequence
                                "max" - Max pool over sequence
                                "cls" - Use first token (BERT-style)
            attention_mask: Mask for valid tokens (1 = valid, 0 = padding)
                           Shape: [batch_size, seq_len]

        Returns:
            Memory embedding to store in cache
            Shape: [batch_size, hidden_dim]

        Implementation (Paper Eq. 4):
            1. Extract representation from backbone output
               - Paper uses last token: h_n = hidden_states[:, -1, :]
               - This captures the final state after processing full context
            2. Apply compression (learnable projection)
               - m_n = compress(h_n)
            3. Return compressed embedding for cache storage

        Extraction Strategies Explained:
            - "last": Last token hidden state (causal LM standard)
              * Best for auto-regressive models (GPT, LLaMA)
              * Captures full left-to-right context
            - "mean": Average over all tokens
              * Good for balanced representation
              * Handles variable-length sequences well
            - "max": Max pooling over tokens
              * Captures salient features
              * Less common but useful for some tasks
            - "cls": First token (BERT-style)
              * For models with special [CLS] tokens
              * Not typical for decoder-only models
        """
        batch_size, seq_len, hidden_dim = backbone_hidden_states.shape

        # Step 1: Extract representation based on strategy (Paper Eq. 4)
        if extraction_strategy == "last":
            # Use last token's hidden state (standard for causal LMs)
            # This is the paper's approach: m_n comes from final position
            if attention_mask is not None:
                # Find actual last non-padded position for each batch element
                # attention_mask: [batch, seq_len], 1 = valid, 0 = padding
                seq_lengths = attention_mask.sum(dim=1)  # [batch]
                # Gather last valid token for each batch element
                indices = (seq_lengths - 1).clamp(min=0)  # [batch]
                indices = indices.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                indices = indices.expand(-1, 1, hidden_dim)  # [batch, 1, hidden_dim]
                representation = torch.gather(backbone_hidden_states, dim=1, index=indices).squeeze(
                    1
                )  # [batch, hidden_dim]
            else:
                # No mask: simply use last token
                representation = backbone_hidden_states[:, -1, :]  # [batch, hidden_dim]

        elif extraction_strategy == "mean":
            # Mean pooling over sequence (masked if attention_mask provided)
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
                masked_hidden = backbone_hidden_states * mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
                sum_mask = mask_expanded.sum(dim=1)  # [batch, 1]
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
                representation = sum_hidden / sum_mask  # [batch, hidden_dim]
            else:
                # Simple mean pooling
                representation = backbone_hidden_states.mean(dim=1)  # [batch, hidden_dim]

        elif extraction_strategy == "max":
            # Max pooling over sequence dimension
            if attention_mask is not None:
                # Mask out padding before max pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
                masked_hidden = backbone_hidden_states * mask_expanded
                # Set masked positions to very negative values before max
                masked_hidden = masked_hidden + (1 - mask_expanded) * (-1e9)
                representation = masked_hidden.max(dim=1)[0]  # [batch, hidden_dim]
            else:
                representation = backbone_hidden_states.max(dim=1)[0]  # [batch, hidden_dim]

        elif extraction_strategy == "cls":
            # Use first token (BERT-style [CLS] token)
            representation = backbone_hidden_states[:, 0, :]  # [batch, hidden_dim]

        else:
            raise ValueError(
                f"Unknown extraction_strategy: {extraction_strategy}. "
                f"Choose from: 'last', 'mean', 'max', 'cls'"
            )

        # Step 2: Apply compression (Paper: compress(...))
        # Learned transformation to create optimal memory representation
        memory_embedding = self.compression(representation)  # [batch, hidden_dim]

        return memory_embedding
