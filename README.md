# Hierarchical Memory Transformer (HMT)

PyTorch implementation of **Hierarchical Memory Transformer for Efficient Long Context Language Processing** ([arXiv:2405.06067](https://arxiv.org/abs/2405.06067)).

HMT is a memory-augmented framework that enables efficient long-context processing by mimicking human memory hierarchy (sensory, short-term, and long-term memory) with a plug-and-play design for any decoder-only transformer.

## Features

- 🧠 **Hierarchical Memory**: Sensory, short-term, and long-term memory mechanisms
- 🔍 **Memory Retrieval**: Cross-attention based relevant context selection
- 🔌 **Plug-and-Play**: Works with any decoder-only transformer (GPT, LLaMA, etc.)
- 🚀 **Apple Silicon Optimized**: Full MPS (Metal Performance Shaders) support for M-series chips
- 📊 **Memory Efficient**: O(L) memory complexity vs O(L²) for standard transformers

## Project Status

🚧 **Phase 1 Complete**: Project setup with Apple Silicon acceleration configured

### Roadmap

- [x] Phase 1: Project setup with uv + Python 3.12 + MPS support
- [ ] Phase 2: Dataset preparation (WikiText-103)
- [ ] Phase 3: Core HMT components implementation
- [ ] Phase 4: Backbone model integration
- [ ] Phase 5: Training infrastructure with BPTT
- [ ] Phase 6: Evaluation and experiments
- [ ] Phase 7: Scaling to larger models

## Setup Instructions

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd /Users/aleks/Downloads/HMT-implementation
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies**:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

4. **Verify Apple Silicon MPS acceleration**:
   ```bash
   python tests/test_mps.py
   ```

   You should see:
   ```
   🎉 All tests passed! MPS is ready for HMT training.
   ```

### Troubleshooting

If MPS is not available:
- Ensure you're running macOS 12.3 or later
- Check PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Reinstall PyTorch with MPS support: `uv pip install --upgrade torch`

## Project Structure

```
HMT-implementation/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── HMT-paper.pdf          # Original research paper
├── src/
│   └── hmt/
│       ├── __init__.py    # Package initialization
│       ├── config.py      # HMT configuration classes
│       ├── model.py       # Main HMT wrapper (Phase 3)
│       ├── memory.py      # Memory components (Phase 3)
│       └── utils.py       # Device utilities with MPS support
├── experiments/
│   ├── configs/           # Experiment configurations
│   └── scripts/           # Training scripts
├── tests/
│   └── test_mps.py       # MPS verification tests
└── data/                  # Dataset cache directory
```

## Usage (Coming Soon)

```python
from hmt import HMT, HMTConfig
from transformers import AutoModelForCausalLM

# Load backbone model
backbone = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure HMT
config = HMTConfig(
    segment_length=512,
    num_memory_embeddings=300,
    sensory_memory_size=32,
)

# Wrap with HMT
model = HMT(backbone, config)

# Process long context (coming in Phase 4-5)
# outputs = model(long_input_ids)
```

## Configuration

Key HMT hyperparameters (from paper):

- `segment_length` (L): Length of each segment (default: 512)
- `representation_length` (j): Tokens used for segment summarization (default: L/2)
- `num_memory_embeddings` (N): Number of cached memory embeddings (default: 300)
- `sensory_memory_size` (k): Tokens preserved from previous segment (default: 32)

## Hardware Requirements

### Minimum (Development)
- Apple Silicon M1/M2/M3/M4 Mac
- 16 GB RAM
- Works with small models (GPT-2, OPT-125M)

### Recommended (Training)
- Apple Silicon M3 Max/Ultra or M4
- 32+ GB RAM
- For larger models (OPT-350M+)

## Paper Reference

```bibtex
@article{he2024hmt,
  title={HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing},
  author={He, Zifan and Cao, Yingqi and Qin, Zongyue and Prakriya, Neha and Sun, Yizhou and Cong, Jason},
  journal={arXiv preprint arXiv:2405.06067},
  year={2024}
}
```

## License

This implementation follows the original paper's research for educational purposes.

## Acknowledgments

- Original HMT paper by He et al. (UCLA, UCSD)
- PyTorch team for MPS backend support
- HuggingFace for transformers library
