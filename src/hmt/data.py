"""
HMT Data Loading Utilities

This module provides data loading for WikiText-103 optimized for HMT training.
Key features:
- Preserves long sequences (no truncation)
- Efficient batching for variable-length articles
- Compatible with HMT's segmentation approach
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Dict, List
import numpy as np


class WikiTextDataset(Dataset):
    """
    WikiText-103 dataset for long-context language modeling with HMT.

    Unlike standard datasets that truncate to max_length, this preserves
    full articles to leverage HMT's hierarchical memory capabilities.

    Args:
        split: Dataset split ('train', 'validation', or 'test')
        tokenizer: HuggingFace tokenizer (e.g., GPT2Tokenizer)
        min_length: Minimum article length in tokens (filters out very short articles)
        max_length: Maximum article length (optional, for memory constraints)
        cache_dir: Directory to cache the dataset
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        min_length: int = 128,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.split = split
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length

        # Load WikiText-103
        print(f"Loading WikiText-103 ({split} split)...")
        self.dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=split,
            cache_dir=cache_dir,
        )

        # Filter and preprocess articles
        self.articles = self._preprocess_articles()

        print(f"Loaded {len(self.articles)} articles from WikiText-103 {split} split")
        if len(self.articles) > 0:
            lengths = [len(article['input_ids']) for article in self.articles]
            print(f"  Token length - Min: {min(lengths)}, Max: {max(lengths)}, "
                  f"Mean: {np.mean(lengths):.1f}")

    def _preprocess_articles(self) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocess and filter articles.

        Returns:
            List of dictionaries with 'input_ids' and 'attention_mask'
        """
        articles = []

        for example in self.dataset:
            text = example['text'].strip()

            # Skip empty or very short articles
            if len(text) == 0:
                continue

            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,  # DON'T truncate - HMT handles long sequences!
                padding=False,
            )

            input_ids = encoding['input_ids'][0]  # Remove batch dimension
            attention_mask = encoding['attention_mask'][0]

            # Filter by length
            if len(input_ids) < self.min_length:
                continue

            if self.max_length is not None and len(input_ids) > self.max_length:
                # Truncate if exceeds max_length
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            articles.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })

        return articles

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single article.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        return self.articles[idx]


def collate_fn_variable_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that pads variable-length sequences.

    HMT processes each article independently, but we still batch them
    for efficiency. This function pads articles to the longest in the batch.

    Args:
        batch: List of dictionaries with 'input_ids' and 'attention_mask'

    Returns:
        Batched and padded tensors
    """
    # Find max length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        # Pad to max_len
        padding_len = max_len - len(input_ids)
        if padding_len > 0:
            # Pad with 0 (typically pad_token_id, but HMT handles this via attention_mask)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(padding_len, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_len, dtype=attention_mask.dtype)
            ])

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
    }


class LongContextDataLoader(DataLoader):
    """
    DataLoader optimized for HMT long-context training.

    This is a thin wrapper around PyTorch's DataLoader with sensible
    defaults for HMT:
    - Variable-length sequences (uses custom collate_fn)
    - Typically small batch sizes (long sequences are memory-intensive)
    - Shuffling enabled for training

    Args:
        dataset: WikiTextDataset instance
        batch_size: Number of articles per batch (default: 4)
        shuffle: Whether to shuffle data (default: True for training)
        num_workers: Number of data loading workers
        **kwargs: Additional arguments passed to DataLoader
    """

    def __init__(
        self,
        dataset: WikiTextDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_variable_length,
            **kwargs
        )


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    min_length: int = 128,
    max_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, LongContextDataLoader]:
    """
    Convenience function to create train/validation/test dataloaders.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for all splits
        min_length: Minimum article length in tokens
        max_length: Maximum article length (None = no limit)
        cache_dir: Cache directory for datasets

    Returns:
        Dictionary with 'train', 'validation', and 'test' DataLoaders

    Example:
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> dataloaders = create_dataloaders(tokenizer, batch_size=2)
        >>> train_loader = dataloaders['train']
        >>> batch = next(iter(train_loader))
        >>> print(batch['input_ids'].shape)  # [batch_size, max_seq_len_in_batch]
    """
    dataloaders = {}

    for split in ['train', 'validation', 'test']:
        dataset = WikiTextDataset(
            split=split,
            tokenizer=tokenizer,
            min_length=min_length,
            max_length=max_length,
            cache_dir=cache_dir,
        )

        dataloaders[split] = LongContextDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Only shuffle training data
        )

    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from hmt.utils import get_device

    print("Testing WikiText data loading...")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=2,
        min_length=128,
        max_length=4096,  # Limit for testing
    )

    # Test train loader
    print("\n" + "=" * 80)
    print("Testing train dataloader:")
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))

    print(f"\nBatch shape:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")

    print(f"\nFirst article in batch:")
    first_article = batch['input_ids'][0]
    # Find actual length (excluding padding)
    actual_len = batch['attention_mask'][0].sum().item()
    print(f"  Actual length: {actual_len} tokens")
    print(f"  First 10 tokens: {first_article[:10].tolist()}")

    # Decode a sample
    print(f"\nDecoded preview (first 100 chars):")
    decoded = tokenizer.decode(first_article[:actual_len])
    print(f"  {decoded[:100]}...")

    # Test on device
    device = get_device()
    print(f"\nMoving to device: {device}")
    batch_on_device = {
        k: v.to(device) for k, v in batch.items()
    }
    print(f"  input_ids device: {batch_on_device['input_ids'].device}")

    print("\n" + "=" * 80)
    print("✅ Data loading test complete!")
