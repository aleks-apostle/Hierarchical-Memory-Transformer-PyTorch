"""
HMT Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HMTConfig:
    """Configuration for Hierarchical Memory Transformer.

    Attributes:
        segment_length (int): Length of each segment (L in paper). Default: 512
        representation_length (int): Number of tokens to use for representation
            encoding (j in paper). Default: 256 (L/2)
        num_memory_embeddings (int): Number of cached memory embeddings (N in paper).
            Default: 300
        sensory_memory_size (int): Number of tokens preserved from previous segment
            (k in paper). Default: 32
        hidden_dim (int): Hidden dimension of the backbone model. Will be auto-detected.
        memory_retrieval (bool): Whether to use memory retrieval mechanism. Default: True
        device (str): Device to use ('mps', 'cuda', or 'cpu'). Auto-detected if None.
    """

    segment_length: int = 512
    representation_length: Optional[int] = None  # Default to segment_length // 2
    num_memory_embeddings: int = 300
    sensory_memory_size: int = 32
    hidden_dim: Optional[int] = None  # Auto-detected from backbone model
    memory_retrieval: bool = True
    device: Optional[str] = None

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.representation_length is None:
            self.representation_length = self.segment_length // 2

        # Validate configuration
        assert self.segment_length > 0, "segment_length must be positive"
        assert self.representation_length <= self.segment_length, \
            "representation_length must be <= segment_length"
        assert self.num_memory_embeddings > 0, "num_memory_embeddings must be positive"
        assert self.sensory_memory_size >= 0, "sensory_memory_size must be non-negative"
        assert self.sensory_memory_size < self.segment_length, \
            "sensory_memory_size must be < segment_length"

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "segment_length": self.segment_length,
            "representation_length": self.representation_length,
            "num_memory_embeddings": self.num_memory_embeddings,
            "sensory_memory_size": self.sensory_memory_size,
            "hidden_dim": self.hidden_dim,
            "memory_retrieval": self.memory_retrieval,
            "device": self.device,
        }
