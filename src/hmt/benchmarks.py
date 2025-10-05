"""
HMT Benchmark Datasets Module

Provides dataset loaders for all benchmarks mentioned in the paper:
- WikiText-103 (already in data.py)
- PG-19 (long-form books)
- arXiv (technical papers)
- Custom long-context benchmarks

Paper Reference: Section 4.1 "Datasets"
- Table 1: WikiText-103, PG-19, arXiv evaluations
- Figure 4: Long-context extrapolation tests
- Table 2: Long-context benchmark results
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import logging

log = logging.getLogger(__name__)


class PG19Dataset(Dataset):
    """
    PG-19 Dataset - Long-form books for long-context evaluation.

    Paper Reference:
      - Table 1: "We evaluate on PG-19 using bits-per-byte (BPB) metric"
      - PG-19 contains 100 test books with average length 69K tokens
      - Tests true long-context understanding

    The PG-19 dataset consists of books from Project Gutenberg published
    before 1919. This is ideal for testing long-context models because:
    1. Very long sequences (10K-100K+ tokens)
    2. Complex narratives requiring long-range understanding
    3. Standard benchmark for long-context LMs

    Example:
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> dataset = PG19Dataset(split='test', tokenizer=tokenizer)
        >>> print(f"Loaded {len(dataset)} books")
        >>> book = dataset[0]
        >>> print(f"Book length: {len(book['input_ids'])} tokens")
    """

    def __init__(
        self,
        split: str = "test",
        tokenizer: PreTrainedTokenizer = None,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize PG-19 dataset.

        Args:
            split: 'train', 'validation', or 'test'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (None = no limit)
            cache_dir: Directory to cache dataset
        """
        super().__init__()

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading PG-19 dataset ({split} split)...")
        try:
            self.dataset = load_dataset(
                "pg19",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            log.warning(f"Failed to load PG-19: {e}")
            log.warning("PG-19 requires manual download. See: https://github.com/deepmind/pg19")
            self.dataset = []
            self.books = []
            return

        # Preprocess books
        self.books = self._preprocess_books()

        log.info(f"Loaded {len(self.books)} books from PG-19 {split} split")
        if len(self.books) > 0:
            lengths = [len(book['input_ids']) for book in self.books]
            log.info(f"  Token length - Min: {min(lengths)}, Max: {max(lengths)}, "
                    f"Mean: {np.mean(lengths):.1f}")

    def _preprocess_books(self) -> List[Dict[str, torch.Tensor]]:
        """Preprocess and tokenize books."""
        books = []

        for example in self.dataset:
            # PG-19 has 'text' field
            text = example.get('text', example.get('book_text', ''))

            if len(text) == 0:
                continue

            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,  # Keep full book
                padding=False,
            )

            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]

            # Apply max_length if specified
            if self.max_length is not None and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            books.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'book_id': example.get('short_book_title', f'book_{len(books)}'),
            })

        return books

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single book."""
        return self.books[idx]


class ArXivDataset(Dataset):
    """
    arXiv Dataset - Technical papers for domain adaptation.

    Paper Reference:
      - Table 1: "arXiv dataset evaluation"
      - Tests model's ability to handle technical/scientific text
      - Long sequences with specialized vocabulary

    arXiv papers are ideal for evaluating:
    1. Technical language understanding
    2. Mathematical notation handling
    3. Long-form scientific reasoning
    4. Domain adaptation capability

    Example:
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> dataset = ArXivDataset(split='test', tokenizer=tokenizer, subject='cs')
        >>> paper = dataset[0]
        >>> print(f"Paper length: {len(paper['input_ids'])} tokens")
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        subject: str = "cs",  # Computer Science
        max_length: Optional[int] = None,
        min_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize arXiv dataset.

        Args:
            split: Dataset split
            tokenizer: HuggingFace tokenizer
            subject: Subject area ('cs', 'math', 'physics', etc.)
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            cache_dir: Cache directory
        """
        super().__init__()

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.split = split
        self.tokenizer = tokenizer
        self.subject = subject
        self.max_length = max_length
        self.min_length = min_length

        log.info(f"Loading arXiv dataset (subject={subject}, split={split})...")
        try:
            # arXiv dataset from HuggingFace
            self.dataset = load_dataset(
                "scientific_papers",
                "arxiv",
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as e:
            log.warning(f"Failed to load arXiv dataset: {e}")
            self.dataset = []
            self.papers = []
            return

        # Preprocess papers
        self.papers = self._preprocess_papers()

        log.info(f"Loaded {len(self.papers)} papers from arXiv {split} split")
        if len(self.papers) > 0:
            lengths = [len(paper['input_ids']) for paper in self.papers]
            log.info(f"  Token length - Min: {min(lengths)}, Max: {max(lengths)}, "
                    f"Mean: {np.mean(lengths):.1f}")

    def _preprocess_papers(self) -> List[Dict[str, torch.Tensor]]:
        """Preprocess and tokenize papers."""
        papers = []

        for example in self.dataset:
            # Combine abstract and article
            abstract = example.get('abstract', '')
            article = example.get('article', '')
            text = f"{abstract}\n\n{article}"

            if len(text) == 0:
                continue

            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,
                padding=False,
            )

            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]

            # Filter by length
            if len(input_ids) < self.min_length:
                continue

            if self.max_length is not None and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            papers.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'paper_id': example.get('article_id', f'paper_{len(papers)}'),
            })

        return papers

    def __len__(self) -> int:
        return len(self.papers)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single paper."""
        return self.papers[idx]


class LongContextBenchmark(Dataset):
    """
    Custom long-context benchmark for Figure 4 reproduction.

    Paper Reference:
      - Figure 4: "Extrapolation to Longer Context"
      - Tests model performance at various sequence lengths
      - Demonstrates HMT maintains PPL while baselines degrade

    This benchmark creates test sequences of specific lengths
    to evaluate long-context extrapolation capability.

    Methodology:
    1. Takes base dataset (WikiText, PG-19, etc.)
    2. Samples/truncates to specific sequence lengths
    3. Evaluates at multiple lengths: 512, 1024, 2048, 4096, 8192
    4. Plots PPL vs sequence length (Figure 4)

    Example (Reproduce Figure 4):
        >>> from hmt.data import WikiTextDataset
        >>> base_dataset = WikiTextDataset(split='test', tokenizer=tokenizer)
        >>>
        >>> # Create benchmarks at different lengths
        >>> benchmarks = {}
        >>> for length in [512, 1024, 2048, 4096, 8192]:
        ...     benchmarks[length] = LongContextBenchmark(
        ...         base_dataset,
        ...         target_length=length,
        ...         num_samples=100
        ...     )
        >>>
        >>> # Evaluate HMT at each length
        >>> results = {}
        >>> for length, bench in benchmarks.items():
        ...     ppl = evaluator.evaluate_perplexity(DataLoader(bench))
        ...     results[length] = ppl
        >>>
        >>> # Plot Figure 4
        >>> plt.plot(list(results.keys()), [r['perplexity'] for r in results.values()])
        >>> plt.xlabel("Sequence Length")
        >>> plt.ylabel("Perplexity")
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_length: int,
        num_samples: Optional[int] = None,
        strategy: str = "truncate",  # 'truncate' or 'sample'
    ):
        """
        Create long-context benchmark at specific length.

        Args:
            base_dataset: Base dataset to sample from
            target_length: Target sequence length
            num_samples: Number of samples (None = all available)
            strategy: 'truncate' (take first N tokens) or
                     'sample' (sample random N-token window)
        """
        super().__init__()

        self.base_dataset = base_dataset
        self.target_length = target_length
        self.strategy = strategy

        log.info(f"Creating long-context benchmark (length={target_length}, "
                f"strategy={strategy})...")

        # Create samples at target length
        self.samples = self._create_samples(num_samples)

        log.info(f"Created {len(self.samples)} samples at length {target_length}")

    def _create_samples(self, num_samples: Optional[int]) -> List[Dict[str, torch.Tensor]]:
        """Create samples at target length."""
        samples = []
        max_samples = num_samples if num_samples is not None else len(self.base_dataset)

        for idx in range(min(max_samples, len(self.base_dataset))):
            item = self.base_dataset[idx]
            input_ids = item['input_ids']
            attention_mask = item.get('attention_mask', torch.ones_like(input_ids))

            # Check if sequence is long enough
            if len(input_ids) < self.target_length:
                continue

            if self.strategy == "truncate":
                # Take first target_length tokens
                sampled_ids = input_ids[:self.target_length]
                sampled_mask = attention_mask[:self.target_length]

            elif self.strategy == "sample":
                # Sample random window of target_length
                if len(input_ids) == self.target_length:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, len(input_ids) - self.target_length + 1, (1,)).item()
                end_idx = start_idx + self.target_length

                sampled_ids = input_ids[start_idx:end_idx]
                sampled_mask = attention_mask[start_idx:end_idx]

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            samples.append({
                'input_ids': sampled_ids,
                'attention_mask': sampled_mask,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        return self.samples[idx]


def create_benchmark_dataloaders(
    tokenizer: PreTrainedTokenizer,
    benchmark_name: str = "wikitext",
    split: str = "test",
    batch_size: int = 1,
    max_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    Convenience function to create benchmark dataloaders.

    Args:
        tokenizer: HuggingFace tokenizer
        benchmark_name: 'wikitext', 'pg19', or 'arxiv'
        split: Dataset split
        batch_size: Batch size
        max_length: Maximum sequence length
        cache_dir: Cache directory

    Returns:
        DataLoader for the specified benchmark

    Example:
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> pg19_loader = create_benchmark_dataloaders(
        ...     tokenizer,
        ...     benchmark_name='pg19',
        ...     split='test',
        ...     batch_size=1
        ... )
    """
    if benchmark_name == "wikitext":
        from .data import WikiTextDataset, collate_fn_variable_length
        dataset = WikiTextDataset(
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
        )

    elif benchmark_name == "pg19":
        from .data import collate_fn_variable_length
        dataset = PG19Dataset(
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
        )

    elif benchmark_name == "arxiv":
        from .data import collate_fn_variable_length
        dataset = ArXivDataset(
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
        )

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Import collate function
    from .data import collate_fn_variable_length

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        collate_fn=collate_fn_variable_length,
    )

    return dataloader


if __name__ == "__main__":
    print("=" * 80)
    print("HMT Benchmark Datasets Module")
    print("=" * 80)

    print("\nAvailable benchmark datasets:")
    print("1. PG19Dataset - Long-form books (Table 1)")
    print("2. ArXivDataset - Technical papers (Table 1)")
    print("3. LongContextBenchmark - Variable-length testing (Figure 4)")

    print("\nUsage example:")
    print("""
    from transformers import GPT2Tokenizer
    from hmt.benchmarks import PG19Dataset, LongContextBenchmark

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Create PG-19 benchmark
    pg19 = PG19Dataset(split='test', tokenizer=tokenizer)

    # Create long-context benchmark at 4096 tokens
    long_bench = LongContextBenchmark(
        base_dataset=pg19,
        target_length=4096,
        num_samples=100
    )
    """)

    print("\n" + "=" * 80)
    print("See class docstrings for detailed usage")
    print("=" * 80)
