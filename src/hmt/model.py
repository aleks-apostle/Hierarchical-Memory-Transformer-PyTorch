"""
HMT Model Wrapper

Main HMT class that wraps a backbone transformer model.

Paper Reference: "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from collections import deque

from .config import HMTConfig
from .memory import RepresentationEncoder, MemorySearch, MemoryEmbeddingGenerator


class HMT(nn.Module):
    """
    Hierarchical Memory Transformer wrapper.

    Paper Reference:
      - Section 3: "HMT Architecture"
      - Figure 1: Complete pipeline overview
      - Algorithm 1: HMT training procedure

    This class wraps any decoder-only transformer model and augments it with
    hierarchical memory capabilities for efficient long-context processing.

    Three-Level Memory Hierarchy:
      1. Sensory Memory (k tokens): Preserves last k tokens from previous segment
      2. Short-term Memory (L tokens): Current segment being processed
      3. Long-term Memory (N embeddings): Cache of memory embeddings from past

    Processing Flow (per segment):
      1. Retrieve sensory memory from previous segment (last k tokens)
      2. Encode current segment → representation S_n (RepresentationEncoder)
      3. Search long-term cache for relevant memories → P_n (MemorySearch)
      4. Create augmented input: [sensory || segment || retrieved_memory]
      5. Process through backbone model
      6. Generate memory embedding m_n (MemoryEmbeddingGenerator)
      7. Update long-term cache with m_n (FIFO queue, max size N)
      8. Update sensory memory (last k tokens of current segment)
    """

    def __init__(self, backbone_model, config: HMTConfig):
        """
        Args:
            backbone_model: Pre-trained decoder-only transformer model
                           (e.g., GPT-2, LLaMA, OPT)
            config: HMT configuration
        """
        super().__init__()
        self.backbone = backbone_model
        self.config = config

        # Auto-detect hidden dimension from backbone
        if config.hidden_dim is None:
            # Try to get from backbone config
            if hasattr(backbone_model, "config"):
                self.config.hidden_dim = backbone_model.config.hidden_size
            else:
                raise ValueError("Could not auto-detect hidden_dim. Please specify in config.")

        # Initialize memory components (Phase 3.2)
        self.representation_encoder = RepresentationEncoder(self.config)
        self.memory_search = MemorySearch(self.config)
        self.memory_embedding_generator = MemoryEmbeddingGenerator(self.config)

        # Memory cache: FIFO queue of memory embeddings (max size N)
        # Each element: [batch_size, hidden_dim]
        self.memory_cache: deque = deque(maxlen=self.config.num_memory_embeddings)

        # Sensory memory: Last k tokens from previous segment
        # Shape: [batch_size, k, hidden_dim] or None (for first segment)
        self.sensory_memory: Optional[torch.Tensor] = None

        # Get embedding layer from backbone for converting retrieved memory
        # to pseudo-token embeddings
        self.embedding_layer = self._get_embedding_layer()

    def _get_embedding_layer(self):
        """
        Extract the embedding layer from backbone model.

        This is used to get hidden dimension and convert input_ids to embeddings.
        Different model architectures have different attribute names.
        """
        # Try common attribute names for embedding layers
        if hasattr(self.backbone, "transformer"):
            # GPT-2 style: model.transformer.wte
            if hasattr(self.backbone.transformer, "wte"):
                return self.backbone.transformer.wte
        elif hasattr(self.backbone, "model"):
            # LLaMA/OPT style: model.model.embed_tokens
            if hasattr(self.backbone.model, "embed_tokens"):
                return self.backbone.model.embed_tokens
        elif hasattr(self.backbone, "embeddings"):
            # Some models: model.embeddings.word_embeddings
            if hasattr(self.backbone.embeddings, "word_embeddings"):
                return self.backbone.embeddings.word_embeddings

        # Fallback: try to find any embedding layer
        for module in self.backbone.modules():
            if isinstance(module, nn.Embedding):
                return module

        raise ValueError(
            "Could not find embedding layer in backbone model. "
            "Please ensure the backbone has an accessible embedding layer."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_memory: bool = True,
    ):
        """
        Forward pass through HMT with hierarchical memory.

        Paper Reference:
          - Section 3: Complete HMT architecture
          - Algorithm 1: Training procedure with BPTT

        Args:
            input_ids: Input token IDs
                      Shape: [batch_size, sequence_length]
            attention_mask: Attention mask (1 = valid, 0 = padding)
                           Shape: [batch_size, sequence_length]
            return_dict: Whether to return a dict (compatible with HuggingFace)
            use_memory: Whether to use memory retrieval (ablation control)

        Returns:
            If return_dict=True: Model output dict with 'logits', 'hidden_states', etc.
            If return_dict=False: Tuple of (logits, hidden_states, ...)

        Processing Algorithm:
            1. Segment input into chunks of length L
            2. For each segment n:
               a. Retrieve sensory memory (last k tokens from segment n-1)
               b. Encode segment → S_n (RepresentationEncoder)
               c. Search cache → P_n (MemorySearch)
               d. Augment: [sensory || segment || P_n as pseudo-token]
               e. Process through backbone → hidden states
               f. Generate memory → m_n (MemoryEmbeddingGenerator)
               g. Update cache with m_n (FIFO, max N)
               h. Save last k tokens for next segment
            3. Concatenate all segment outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get input embeddings from backbone
        # Shape: [batch_size, seq_len, hidden_dim]
        input_embeddings = self.embedding_layer(input_ids)

        # Segment the input into chunks of length L
        segment_length = self.config.segment_length
        num_segments = (seq_len + segment_length - 1) // segment_length  # Ceiling division

        # Storage for outputs from all segments
        all_logits = []
        all_hidden_states = []

        # Process each segment
        for segment_idx in range(num_segments):
            # Extract current segment
            start_idx = segment_idx * segment_length
            end_idx = min(start_idx + segment_length, seq_len)

            # Current segment embeddings and mask
            segment_embeddings = input_embeddings[:, start_idx:end_idx, :]
            segment_mask = (
                attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
            )

            # === STEP 1: Representation Encoding (Paper Section 3.2) ===
            # Encode first j tokens of segment into summary S_n
            with torch.no_grad():  # Don't backprop through encoder during inference
                segment_representation = self.representation_encoder(
                    segment_embeddings, attention_mask=segment_mask
                )  # [batch, hidden_dim]

            # === STEP 2: Memory Search (Paper Section 3.2, Eq. 2-3) ===
            # Retrieve relevant memory P_n from long-term cache
            if use_memory and len(self.memory_cache) > 0:
                # Stack cached memories: [batch, cache_size, hidden_dim]
                cached_memories = torch.stack(list(self.memory_cache), dim=1)

                # Search for relevant memory
                with torch.no_grad():  # Don't backprop through search during inference
                    retrieved_memory = self.memory_search(
                        segment_representation, cached_memories
                    )  # [batch, hidden_dim]
            else:
                # No cache yet (first segment) or memory disabled
                retrieved_memory = torch.zeros(
                    batch_size, self.config.hidden_dim, device=device, dtype=input_embeddings.dtype
                )

            # === STEP 3: Create Augmented Input (Paper Section 3.3) ===
            # Augmented input: [sensory_memory || segment || retrieved_memory]

            augmented_embeddings = []

            # Add sensory memory (last k tokens from previous segment)
            if self.sensory_memory is not None and use_memory:
                augmented_embeddings.append(self.sensory_memory)

            # Add current segment
            augmented_embeddings.append(segment_embeddings)

            # Add retrieved memory as a pseudo-token
            # Shape: [batch, 1, hidden_dim]
            if use_memory:
                retrieved_memory_token = retrieved_memory.unsqueeze(1)
                augmented_embeddings.append(retrieved_memory_token)

            # Concatenate all parts: [batch, augmented_length, hidden_dim]
            augmented_input = torch.cat(augmented_embeddings, dim=1)

            # Create augmented attention mask
            if segment_mask is not None:
                augmented_mask_parts = []

                if self.sensory_memory is not None and use_memory:
                    # Sensory memory is always valid (all 1s)
                    sensory_mask = torch.ones(
                        batch_size,
                        self.config.sensory_memory_size,
                        device=device,
                        dtype=segment_mask.dtype,
                    )
                    augmented_mask_parts.append(sensory_mask)

                augmented_mask_parts.append(segment_mask)

                if use_memory:
                    # Retrieved memory token is always valid
                    memory_token_mask = torch.ones(
                        batch_size, 1, device=device, dtype=segment_mask.dtype
                    )
                    augmented_mask_parts.append(memory_token_mask)

                augmented_mask = torch.cat(augmented_mask_parts, dim=1)
            else:
                augmented_mask = None

            # === STEP 4: Process Through Backbone (Paper: BBM) ===
            # Pass augmented input through backbone model
            backbone_outputs = self.backbone(
                inputs_embeds=augmented_input,
                attention_mask=augmented_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract outputs corresponding to the current segment
            # (ignore sensory and memory token outputs)
            sensory_length = (
                self.config.sensory_memory_size
                if (self.sensory_memory is not None and use_memory)
                else 0
            )
            segment_start = sensory_length
            segment_end = segment_start + (end_idx - start_idx)

            # Logits for current segment
            segment_logits = backbone_outputs.logits[:, segment_start:segment_end, :]
            all_logits.append(segment_logits)

            # Hidden states for current segment
            segment_hidden = backbone_outputs.hidden_states[-1][:, segment_start:segment_end, :]
            all_hidden_states.append(segment_hidden)

            # === STEP 5: Generate Memory Embedding (Paper Section 3.3, Eq. 4) ===
            # Create memory embedding m_n from augmented segment output
            if use_memory:
                full_hidden_states = backbone_outputs.hidden_states[-1]  # Last layer
                memory_embedding = self.memory_embedding_generator(
                    full_hidden_states,
                    extraction_strategy="last",
                    attention_mask=augmented_mask,
                )  # [batch, hidden_dim]

                # === STEP 6: Update Long-term Memory Cache ===
                # Add to cache (FIFO queue, auto-evicts oldest if exceeds N)
                self.memory_cache.append(memory_embedding.detach())  # Detach to save memory

            # === STEP 7: Update Sensory Memory ===
            # Save last k tokens from current segment for next iteration
            if use_memory:
                k = self.config.sensory_memory_size
                if segment_embeddings.size(1) >= k:
                    # Take last k tokens
                    self.sensory_memory = segment_embeddings[:, -k:, :].detach()
                else:
                    # Segment is shorter than k, use all available
                    self.sensory_memory = segment_embeddings.detach()

        # === STEP 8: Concatenate All Segment Outputs ===
        # Combine logits and hidden states from all segments
        full_logits = torch.cat(all_logits, dim=1)  # [batch, seq_len, vocab_size]
        full_hidden_states = torch.cat(all_hidden_states, dim=1)  # [batch, seq_len, hidden_dim]

        # Return in HuggingFace format
        if return_dict:
            return {
                "logits": full_logits,
                "hidden_states": (full_hidden_states,),
                "past_key_values": None,  # Not used in HMT
            }
        else:
            return (full_logits, full_hidden_states)

    def clear_memory(self):
        """
        Clear all memory components.

        Use this between different documents/sequences to prevent
        cross-contamination of memory across independent contexts.
        """
        self.memory_cache.clear()
        self.sensory_memory = None

    def get_memory_stats(self) -> dict:
        """
        Get statistics about current memory state.

        Returns:
            Dict with memory cache size, sensory memory info, etc.
        """
        return {
            "cache_size": len(self.memory_cache),
            "max_cache_size": self.config.num_memory_embeddings,
            "sensory_memory_active": self.sensory_memory is not None,
            "sensory_memory_size": (
                self.sensory_memory.shape[1] if self.sensory_memory is not None else 0
            ),
            "segment_length": self.config.segment_length,
        }
