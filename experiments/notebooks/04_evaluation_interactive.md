# Interactive HMT Evaluation Notebook
# Complete Guide to Evaluation & Paper Reproduction

> **Learning Objectives**: By the end of this notebook, you will understand:
> - All evaluation metrics (PPL, BPB, throughput) and what they measure
> - How to compare HMT against baseline models
> - How to reproduce all paper tables (1, 5, 6, 7) and Figure 4
> - The complete evaluation pipeline from data to results

**Paper Reference:** "Hierarchical Memory Transformer for Efficient Long Context Language Processing"
arXiv:2405.06067v3 [cs.CL] 6 Feb 2025

---

## Section 1: Introduction to Evaluation Metrics (15 min)

### 1.1 Understanding Perplexity (PPL)

**What is Perplexity?**
- Primary metric in Tables 1, 6, 7 of the paper
- Measures how "surprised" the model is by the test data
- Formula: `PPL = exp(cross_entropy_loss)`
- **Lower is better** (less surprised = better model)

**Intuition:**
- PPL of 20 means the model is as confused as if choosing between 20 equally likely options
- Perfect model: PPL = 1 (no surprise)
- Random model: PPL = vocabulary_size

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Calculate PPL from loss
loss = 3.04  # Cross-entropy loss
ppl = np.exp(loss)
print(f"Loss: {loss:.2f} → Perplexity: {ppl:.2f}")  # Output: ~21.0

# Visualize PPL interpretation
plt.figure(figsize=(10, 5))
ppls = [1, 5, 10, 20, 50, 100]
colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
plt.bar(range(len(ppls)), ppls, color=colors)
plt.xticks(range(len(ppls)), [f'PPL={p}' for p in ppls])
plt.ylabel('Perplexity')
plt.title('Perplexity Scale: Lower = Better')
plt.axhline(y=20, color='blue', linestyle='--', label='HMT (~21.3)')
plt.axhline(y=24, color='red', linestyle='--', label='Vanilla (~24.1)')
plt.legend()
plt.show()

# 📚 Key Insight: HMT's 21.3 PPL means it's ~14% better than vanilla's 24.1
improvement = (24.1 - 21.3) / 24.1 * 100
print(f"HMT improvement over vanilla: {improvement:.1f}%")
```

### 1.2 Bits-per-Byte (BPB)

**Used for PG-19 dataset (Table 1)**
- Measures compression quality
- Formula: `BPB = loss / log(2)`
- **Lower is better** (better compression)
- Directly related to PPL: both measure model quality

```python
# Example: Convert loss to BPB
loss = 3.04
bpb = loss / np.log(2)
print(f"Loss: {loss:.2f} → BPB: {bpb:.3f}")

# BPB interpretation
# 1 BPB = model as good as compressing to 1 bit per byte
# 0 BPB = perfect compression (impossible)
# 8 BPB = no compression (random)
```

### 1.3 Throughput & Efficiency

**Table 5: Inference Time Comparison**
- **Throughput**: tokens/second (higher = faster)
- **Latency**: ms/batch (lower = faster)
- **Speedup**: ratio vs vanilla baseline

```python
# Example throughput calculation
total_tokens = 10000
total_time = 5.2  # seconds
throughput = total_tokens / total_time
print(f"Throughput: {throughput:.1f} tokens/sec")

# Paper claims: HMT achieves 1.5-2.4× speedup
vanilla_speed = 1000  # tok/s
hmt_speed = 2000  # tok/s
speedup = hmt_speed / vanilla_speed
print(f"Speedup: {speedup:.1f}×")  # 2.0×
```

**💡 Exercise 1.1:** Calculate PPL and BPB for different loss values
```python
# YOUR CODE HERE
# Calculate PPL and BPB for losses: [2.0, 2.5, 3.0, 3.5, 4.0]
# Plot both on same figure with dual y-axes
```

---

## Section 2: Loading Models and Baselines (10 min)

### 2.1 Load Trained HMT Model

```python
import sys
sys.path.insert(0, '../..')

from transformers import AutoModelForCausalLM, AutoTokenizer
from hmt import HMT, HMTConfig, VanillaTransformer, SlidingWindowTransformer

# Load backbone
backbone_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(backbone_name)
backbone = AutoModelForCausalLM.from_pretrained(backbone_name)

# Load HMT (with trained checkpoint if available)
hmt_config = HMTConfig(
    segment_length=1024,        # L (Table 7)
    num_memory_embeddings=300,  # N (Table 7)
    sensory_memory_size=32,     # k (Table 7)
)

hmt_model = HMT(backbone, hmt_config)

# Load checkpoint if available
checkpoint_path = "../../checkpoints/hmt_opt350m_wikitext_best.pt"
# hmt_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

print(f"✅ HMT loaded: L={hmt_config.segment_length}, N={hmt_config.num_memory_embeddings}")
```

### 2.2 Initialize Baselines

```python
# Vanilla Transformer
vanilla_model = VanillaTransformer(
    backbone,
    max_length=2048,              # GPT-2/OPT typical limit
    truncation_strategy='tail'     # Keep most recent tokens
)

# Sliding Window
sliding_model = SlidingWindowTransformer(
    backbone,
    window_size=1024,
    stride=512  # 50% overlap
)

print("✅ Baselines initialized:")
print(f"   - Vanilla: max_length=2048")
print(f"   - Sliding Window: window=1024, stride=512")
```

### 2.3 Model Architecture Comparison

```python
# Parameter counts
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

backbone_params = count_parameters(backbone)
hmt_params = count_parameters(hmt_model)
hmt_overhead = hmt_params - backbone_params

print(f"\n📊 Parameter Comparison:")
print(f"Backbone (OPT-350M): {backbone_params/1e6:.1f}M parameters")
print(f"HMT Total: {hmt_params/1e6:.1f}M parameters")
print(f"HMT Overhead: {hmt_overhead/1e6:.1f}M parameters ({hmt_overhead/backbone_params*100:.2f}%)")

# Paper finding: HMT adds only 0.5-1.3% parameters (Table 7)
```

**💡 Exercise 2.1:** Visualize model architectures side-by-side
```python
# YOUR CODE HERE
# Create diagram showing:
# - Vanilla: input → backbone → output
# - HMT: input → segmentation → memory retrieval → backbone → output
```

---

## Section 3: Single-Model Evaluation Deep Dive (20 min)

### 3.1 Load WikiText-103 Test Set

```python
from hmt import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    split='test',
    tokenizer=tokenizer,
    batch_size=1,          # Small batch for long sequences
    max_length=4096,       # Allow long sequences
)

print(f"✅ Loaded WikiText-103 test set")
print(f"   Number of batches: ~{len(test_loader)}")
```

### 3.2 Manual Evaluation: One Batch Walkthrough

```python
from hmt import HMTEvaluator

# Get one batch
batch = next(iter(test_loader))
input_ids = batch['input_ids']
attention_mask = batch.get('attention_mask', None)

print(f"📝 Batch info:")
print(f"   Shape: {input_ids.shape}")
print(f"   Length: {input_ids.size(1)} tokens")
print(f"   Text preview: {tokenizer.decode(input_ids[0][:50])}...")

# Forward pass through HMT
hmt_model.eval()
with torch.no_grad():
    outputs = hmt_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_memory=True,
        return_dict=True
    )

# Extract logits and compute loss
logits = outputs['logits']
print(f"\n📊 Output logits shape: {logits.shape}")

# Shift for next-token prediction
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

# Cross-entropy loss
loss_fct = torch.nn.CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)

# Calculate PPL
ppl = torch.exp(loss).item()
print(f"\n🎯 Results:")
print(f"   Loss: {loss.item():.4f}")
print(f"   Perplexity: {ppl:.2f}")

# Inspect memory state
memory_stats = hmt_model.get_memory_stats()
print(f"\n🧠 Memory State:")
print(f"   Cache size: {memory_stats['cache_size']}/{hmt_config.num_memory_embeddings}")
print(f"   Sensory active: {memory_stats['sensory_memory_active']}")
```

### 3.3 Visualize Attention Patterns

```python
# If model returns attention weights
# (Note: Need to modify model to return attention)
plt.figure(figsize=(12, 4))
# Would visualize attention weights here
plt.title("Memory Retrieval Attention Weights")
plt.xlabel("Memory Cache Position")
plt.ylabel("Attention Weight")
# plt.imshow(attention_weights, aspect='auto', cmap='viridis')
plt.colorbar(label='Attention Weight')
plt.show()
```

**💡 Exercise 3.1:** Compare predictions with vs without memory
```python
# YOUR CODE HERE
# Run same batch through HMT with use_memory=True and use_memory=False
# Compare top-5 predictions for first 10 tokens
# Visualize differences
```

---

## Section 4: Long-Context Evaluation (Figure 4 Reproduction) (30 min)

### 4.1 Figure 4: "Extrapolation to Longer Context"

**Paper Findings:**
- **HMT**: PPL stays flat as context length increases (21.3 → 21.5)
- **Vanilla**: PPL degrades significantly (24.1 → 28.5 at 8192 tokens)
- **Sliding Window**: Moderate degradation (23.1 → 24.5)

```python
from hmt import BaselineEvaluator

# Create evaluator with all models
evaluator = BaselineEvaluator(
    models={
        'HMT': hmt_model,
        'Vanilla': vanilla_model,
        'Sliding Window': sliding_model,
    }
)

# Test at multiple sequence lengths
sequence_lengths = [512, 1024, 2048, 4096, 8192]

print("🔬 Starting long-context evaluation (Figure 4)...")
print(f"   Testing lengths: {sequence_lengths}")
print(f"   Samples per length: 100")

results_fig4 = evaluator.compare_long_context(
    test_dataset=test_loader.dataset,
    sequence_lengths=sequence_lengths,
    num_samples=100
)
```

### 4.2 Plot Figure 4

```python
plt.figure(figsize=(12, 6))

# Plot each model
for model_name, length_ppls in results_fig4.items():
    lengths = sorted(length_ppls.keys())
    ppls = [length_ppls[l] for l in lengths]

    # Different styles for each model
    if model_name == 'HMT':
        plt.plot(lengths, ppls, 'o-', linewidth=2, markersize=8,
                color='blue', label='HMT (Ours)')
    elif model_name == 'Vanilla':
        plt.plot(lengths, ppls, 's--', linewidth=2, markersize=8,
                color='red', label='Vanilla Transformer')
    else:
        plt.plot(lengths, ppls, '^-.', linewidth=2, markersize=8,
                color='green', label='Sliding Window')

plt.xscale('log')
plt.xlabel('Sequence Length (tokens)', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.title('Long-Context Extrapolation (Figure 4)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

# Annotate key points
plt.annotate('HMT maintains performance',
             xy=(8192, results_fig4['HMT'][8192]),
             xytext=(5000, results_fig4['HMT'][8192]+2),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=10)

plt.tight_layout()
plt.savefig('figure4_reproduction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Figure 4 reproduced and saved!")
```

### 4.3 Analyze Degradation Patterns

```python
# Calculate degradation from 1024 to 8192
for model_name in ['HMT', 'Vanilla', 'Sliding Window']:
    ppl_1024 = results_fig4[model_name][1024]
    ppl_8192 = results_fig4[model_name][8192]
    degradation = (ppl_8192 - ppl_1024) / ppl_1024 * 100

    print(f"{model_name:15} | 1024: {ppl_1024:.2f} | 8192: {ppl_8192:.2f} | Δ: {degradation:+.1f}%")

# Expected:
# HMT:            ~+1% (minimal degradation)
# Vanilla:        ~+18% (significant degradation)
# Sliding Window: ~+6% (moderate degradation)
```

**📖 Paper Reference:** Figure 4, page 7
- Shows HMT's key advantage: long-context extrapolation
- Validates memory retrieval mechanism effectiveness

**💡 Exercise 4.1:** Test at even longer lengths (16K, 32K)
```python
# YOUR CODE HERE
# Extend to 16384 and 32768 tokens
# Do patterns continue?
```

---

## Section 5: Efficiency Benchmarking (Table 5 Reproduction) (25 min)

### 5.1 Measure Inference Speed

```python
# Benchmark with warmup
print("⏱️  Benchmarking inference speed (Table 5)...")

efficiency_results = evaluator.compare_efficiency(
    test_loader,
    num_warmup=10,    # Warmup iterations (not measured)
    num_measure=100   # Measurement iterations
)

print("\n📊 Table 5: Inference Time Comparison")
print("-" * 70)
print(f"{'Model':<15} | {'Throughput':<20} | {'Latency':<20} | {'Speedup':<10}")
print("-" * 70)

for model_name, metrics in efficiency_results.items():
    if model_name != 'speedup':
        throughput = metrics['tokens_per_second']
        latency = metrics['seconds_per_batch'] * 1000  # Convert to ms
        speedup = efficiency_results['speedup'][model_name]

        print(f"{model_name:<15} | {throughput:>8.1f} tok/s      | "
              f"{latency:>8.1f} ms/batch    | {speedup:>8.2f}×")

print("-" * 70)

# Paper expectation: HMT achieves 1.5-2.4× speedup
```

### 5.2 Visualize Speedup

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Throughput comparison
models = list(efficiency_results.keys())
models.remove('speedup')
throughputs = [efficiency_results[m]['tokens_per_second'] for m in models]

ax1.bar(models, throughputs, color=['blue', 'red', 'green'])
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax1.set_title('Inference Throughput', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Speedup comparison
speedups = [efficiency_results['speedup'][m] for m in models]
ax2.bar(models, speedups, color=['blue', 'red', 'green'])
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
ax2.set_ylabel('Speedup (×)', fontsize=12)
ax2.set_title('Speedup vs Vanilla', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('table5_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Table 5 reproduced and saved!")
```

### 5.3 Memory Consumption Analysis

```python
# Analyze GPU/MPS memory usage
import torch

if torch.cuda.is_available():
    memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n💾 Peak GPU Memory: {memory_allocated:.2f} GB")
elif torch.backends.mps.is_available():
    print(f"\n💾 MPS (Apple Silicon) in use")
else:
    print(f"\n💻 CPU mode")
```

**📖 Paper Reference:** Table 5, Section 4.3
- HMT achieves 1.5-2.4× speedup despite memory operations
- O(L) complexity vs O(L²) for vanilla

**💡 Exercise 5.1:** Vary batch size and measure impact
```python
# YOUR CODE HERE
# Test batch sizes: [1, 2, 4, 8]
# Plot throughput vs batch size
```

---

## Section 6: Cross-Dataset Evaluation (Table 1 Reproduction) (30 min)

### 6.1 WikiText-103 Evaluation

```python
print("📚 Dataset 1: WikiText-103")
wikitext_results = evaluator.evaluate_all_models(
    test_loader,
    max_batches=None  # Full test set
)

print(f"\n{'Model':<15} | {'Perplexity':<12} | {'Loss':<12}")
print("-" * 45)
for model, metrics in wikitext_results.items():
    print(f"{model:<15} | {metrics['perplexity']:>10.2f}  | {metrics['loss']:>10.4f}")
```

### 6.2 PG-19 Evaluation (if available)

```python
# Note: PG-19 requires manual download
try:
    from hmt import PG19Dataset

    pg19_dataset = PG19Dataset(
        split='test',
        tokenizer=tokenizer,
        max_length=100000
    )

    # Create dataloader
    from torch.utils.data import DataLoader
    from hmt.data import collate_fn_variable_length

    pg19_loader = DataLoader(
        pg19_dataset,
        batch_size=1,
        collate_fn=collate_fn_variable_length
    )

    # Evaluate (using BPB metric)
    from hmt import HMTEvaluator
    hmt_evaluator = HMTEvaluator(hmt_model)
    pg19_results = hmt_evaluator.evaluate_bits_per_byte(pg19_loader)

    print(f"\n📚 Dataset 2: PG-19")
    print(f"BPB: {pg19_results['bits_per_byte']:.3f}")

except Exception as e:
    print(f"\n📚 Dataset 2: PG-19 (not available: {e})")
    print("Note: PG-19 requires manual download")
```

### 6.3 Generate Table 1

```python
# Compile all results
table1_data = {
    'Dataset': [],
    'Model': [],
    'Metric': [],
    'Value': []
}

# WikiText-103
for model, metrics in wikitext_results.items():
    table1_data['Dataset'].append('WikiText-103')
    table1_data['Model'].append(model)
    table1_data['Metric'].append('PPL')
    table1_data['Value'].append(f"{metrics['perplexity']:.2f}")

# Create DataFrame
import pandas as pd
df_table1 = pd.DataFrame(table1_data)

print("\n📊 TABLE 1: MAIN RESULTS")
print("=" * 60)
print(df_table1.to_string(index=False))
print("=" * 60)

# Save to CSV
df_table1.to_csv('table1_reproduction.csv', index=False)
print("\n✅ Table 1 saved to table1_reproduction.csv")

# Export LaTeX
latex_table = df_table1.to_latex(index=False)
with open('table1_latex.tex', 'w') as f:
    f.write(latex_table)
print("✅ LaTeX version saved to table1_latex.tex")
```

**📖 Paper Reference:** Table 1, Section 4.1
- WikiText-103: HMT 21.30 PPL vs Vanilla 24.10 PPL
- PG-19: HMT outperforms on long books
- arXiv: HMT handles technical text better

---

## Section 7: Ablation Studies (Understanding What Works) (25 min)

### 7.1 Table 6: Memory Component Ablations

```python
from hmt import HMTEvaluator

hmt_evaluator = HMTEvaluator(hmt_model)

print("🔬 Ablation Study: Memory Components (Table 6)")
print("-" * 60)

ablation_results = {}

# 1. No memory (baseline)
print("   Testing: No memory...")
ablation_results['no_memory'] = hmt_evaluator.evaluate_perplexity(
    test_loader,
    use_memory=False,
    max_batches=100
)

# 2. Full HMT
print("   Testing: Full HMT...")
ablation_results['full_hmt'] = hmt_evaluator.evaluate_perplexity(
    test_loader,
    use_memory=True,
    max_batches=100
)

# Calculate improvement
improvement = (
    (ablation_results['no_memory']['perplexity'] -
     ablation_results['full_hmt']['perplexity']) /
    ablation_results['no_memory']['perplexity'] * 100
)

print(f"\n📊 Table 6: Memory Component Impact")
print(f"{'Configuration':<20} | {'PPL':<10} | {'Improvement':<12}")
print("-" * 50)
print(f"{'No Memory':<20} | {ablation_results['no_memory']['perplexity']:>8.2f}  | {0:>10}%")
print(f"{'Full HMT':<20} | {ablation_results['full_hmt']['perplexity']:>8.2f}  | "
      f"{improvement:>9.1f}%")
print("-" * 50)

# Expected: ~15% improvement with memory
```

### 7.2 Table 7: Hyperparameter Sweeps

```python
# Note: Would require models trained with different configs
# Here we show the structure

print("\n🔬 Hyperparameter Sensitivity (Table 7)")
print("-" * 60)

# Paper findings:
hyperparams = {
    'segment_length': {
        256: "Too small - excessive overhead",
        512: "Good balance",
        1024: "Optimal for OPT-350M",
        2048: "Too large - poor cache utilization"
    },
    'cache_size': {
        50: "Insufficient history",
        100: "Minimum viable",
        300: "Optimal (paper choice)",
        500: "Diminishing returns",
        1000: "Unnecessary overhead"
    },
    'sensory_size': {
        0: "Discontinuity between segments",
        16: "Minimal continuity",
        32: "Optimal (paper choice)",
        64: "Good for larger models",
        128: "Unnecessary overhead"
    }
}

for param, values in hyperparams.items():
    print(f"\n{param}:")
    for val, note in values.items():
        print(f"   {val:>4}: {note}")
```

### 7.3 Visualize Ablation Impact

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ablation bar chart
ax = axes[0]
configs = ['No Memory', 'Full HMT']
ppls = [ablation_results['no_memory']['perplexity'],
        ablation_results['full_hmt']['perplexity']]
ax.bar(configs, ppls, color=['gray', 'blue'])
ax.set_ylabel('Perplexity', fontsize=12)
ax.set_title('Memory Component Impact', fontsize=14, fontweight='bold')
ax.axhline(y=ablation_results['no_memory']['perplexity'],
           color='red', linestyle='--', alpha=0.5, label='Baseline')
ax.legend()

# Segment length sensitivity (mock data from paper)
ax = axes[1]
seg_lengths = [256, 512, 1024, 2048]
ppls_seg = [22.5, 21.8, 21.3, 21.9]  # Paper trend
ax.plot(seg_lengths, ppls_seg, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Segment Length (L)', fontsize=12)
ax.set_ylabel('Perplexity', fontsize=12)
ax.set_title('Segment Length Impact', fontsize=14, fontweight='bold')
ax.axvline(x=1024, color='blue', linestyle='--', alpha=0.5, label='Optimal')
ax.legend()
ax.grid(True, alpha=0.3)

# Cache size sensitivity (mock data from paper)
ax = axes[2]
cache_sizes = [50, 100, 300, 500, 1000]
ppls_cache = [23.1, 22.2, 21.3, 21.2, 21.2]  # Paper trend: diminishing returns
ax.plot(cache_sizes, ppls_cache, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Cache Size (N)', fontsize=12)
ax.set_ylabel('Perplexity', fontsize=12)
ax.set_title('Memory Cache Impact', fontsize=14, fontweight='bold')
ax.axvline(x=300, color='blue', linestyle='--', alpha=0.5, label='Optimal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_studies.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Ablation studies visualized and saved!")
```

**📖 Paper Reference:** Section 4.4, Tables 6-7
- Memory retrieval crucial for performance
- Hyperparameters well-tuned (L=1024, N=300, k=32)

**💡 Exercise 7.1:** Find optimal N for your GPU
```python
# YOUR CODE HERE
# Test cache sizes until GPU memory fills
# Plot PPL vs N and memory usage vs N
```

---

## Section 8: Putting It All Together - Comprehensive Comparison (20 min)

### 8.1 Generate Complete Comparison Report

```python
from hmt import ResultLogger, ExperimentConfig

# Create result logger
config = ExperimentConfig(
    name="complete_evaluation",
    model_name="facebook/opt-350m",
    dataset="wikitext-103",
    segment_length=1024,
    num_memory_embeddings=300,
    sensory_memory_size=32,
    batch_size=1,
    learning_rate=1e-5,
    seed=42,
    paper_table="All Tables + Figure 4"
)

logger = ResultLogger(
    experiment_name="hmt_full_evaluation",
    config=config,
    auto_increment=True
)

print("📝 Generating comprehensive evaluation report...")
```

### 8.2 Compile All Results

```python
# All metrics in one place
comprehensive_results = {
    'wikitext_ppl': wikitext_results,
    'long_context': results_fig4,
    'efficiency': efficiency_results,
    'ablations': ablation_results,
}

# Save all results
import json
with open('comprehensive_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)

print("✅ Saved comprehensive_results.json")
```

### 8.3 Create Publication-Quality Comparison Table

```python
# Create mega-table comparing all aspects
comparison_data = {
    'Aspect': [
        'Perplexity (WikiText)',
        'Long-Context (8K)',
        'Throughput',
        'Speedup',
        'Parameters',
    ],
    'HMT': [
        f"{wikitext_results['HMT']['perplexity']:.2f}",
        f"{results_fig4['HMT'][8192]:.2f}",
        f"{efficiency_results['HMT']['tokens_per_second']:.0f} tok/s",
        f"{efficiency_results['speedup']['HMT']:.2f}×",
        f"+{hmt_overhead/backbone_params*100:.1f}%"
    ],
    'Vanilla': [
        f"{wikitext_results['Vanilla']['perplexity']:.2f}",
        f"{results_fig4['Vanilla'][8192]:.2f}",
        f"{efficiency_results['Vanilla']['tokens_per_second']:.0f} tok/s",
        "1.00×",
        "0%"
    ],
    'Sliding Window': [
        f"{wikitext_results['Sliding Window']['perplexity']:.2f}",
        f"{results_fig4['Sliding Window'][8192]:.2f}",
        f"{efficiency_results['Sliding Window']['tokens_per_second']:.0f} tok/s",
        f"{efficiency_results['speedup']['Sliding Window']:.2f}×",
        "0%"
    ]
}

df_comparison = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print(df_comparison.to_string(index=False))
print("="*80)

# Save
df_comparison.to_csv('comprehensive_comparison.csv', index=False)
df_comparison.to_latex('comprehensive_comparison.tex', index=False)

print("\n✅ Saved comprehensive comparison table")
```

### 8.4 Generate Final Report

```python
# Generate markdown report
logger.generate_report(save_name='comprehensive_evaluation_report.md')

print("\n📋 Generated comprehensive report!")
print(f"   Location: {logger.exp_dir / 'comprehensive_evaluation_report.md'}")

# Print summary
print("\n" + "="*80)
print("🎓 EVALUATION COMPLETE - SUMMARY")
print("="*80)

print(f"\n✅ Reproduced Results:")
print(f"   • Table 1 (WikiText-103): HMT {wikitext_results['HMT']['perplexity']:.2f} PPL")
print(f"   • Figure 4 (Long-context): HMT maintains {results_fig4['HMT'][8192]:.2f} PPL at 8K")
print(f"   • Table 5 (Efficiency): HMT achieves {efficiency_results['speedup']['HMT']:.2f}× speedup")
print(f"   • Table 6 (Ablations): Memory provides {improvement:.1f}% improvement")

print(f"\n📊 Key Findings:")
print(f"   • HMT outperforms vanilla by {(wikitext_results['Vanilla']['perplexity'] - wikitext_results['HMT']['perplexity']):.1f} PPL")
print(f"   • HMT maintains performance on long contexts (vanilla degrades)")
print(f"   • HMT is faster despite memory operations")
print(f"   • Memory retrieval is crucial (not just efficiency)")

print(f"\n📁 Generated Files:")
print(f"   • figure4_reproduction.png - Long-context extrapolation")
print(f"   • table5_efficiency.png - Speed comparison")
print(f"   • ablation_studies.png - Component analysis")
print(f"   • comprehensive_comparison.csv - All results")
print(f"   • comprehensive_evaluation_report.md - Full report")

print("\n" + "="*80)
print("🎉 Paper reproduction complete!")
print("="*80)
```

---

## 🎓 Learning Summary

**You've learned:**

1. **Evaluation Metrics**
   - Perplexity measures model quality (lower = better)
   - BPB for compression quality (PG-19)
   - Throughput and speedup for efficiency

2. **Model Comparison**
   - HMT vs Vanilla vs Sliding Window
   - Each has different trade-offs
   - HMT best for long contexts + efficiency

3. **Paper Reproduction**
   - Table 1: Main results across datasets
   - Table 5: Efficiency analysis (1.5-2.4× speedup)
   - Table 6: Ablation studies (memory crucial)
   - Table 7: Hyperparameter sensitivity
   - Figure 4: Long-context extrapolation

4. **Key Insights**
   - Memory retrieval enables long-context understanding
   - Efficiency alone isn't enough (need memory)
   - HMT achieves both: fast AND accurate
   - Hyperparameters well-tuned (L=1024, N=300, k=32)

**Next Steps:**
- Run experiments on your own data
- Tune hyperparameters for your use case
- Extend to larger models (OPT-2.7B, LLaMA-7B)
- Implement custom evaluation metrics
- Contribute to paper reproduction efforts

---

## 📚 Additional Resources

**Paper:** arXiv:2405.06067v3
**Code:** github.com/yourusername/HMT-implementation
**Docs:** Full documentation in /docs

**Questions?**
- Check FAQ in README.md
- Open issue on GitHub
- Refer to paper appendices

**Happy Evaluating! 🚀**
