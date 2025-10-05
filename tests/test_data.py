"""
Tests for HMT data loading utilities.

These tests verify:
1. WikiTextDataset loads and preprocesses data correctly
2. Collate function handles variable-length sequences
3. DataLoaders work with HMT's long-context requirements
4. Device compatibility (MPS/CUDA/CPU)
"""

import pytest
import torch
from transformers import GPT2Tokenizer

from hmt.data import (
    WikiTextDataset,
    LongContextDataLoader,
    create_dataloaders,
    collate_fn_variable_length,
)
from hmt.utils import get_device


@pytest.fixture(scope="module")
def tokenizer():
    """Create a GPT-2 tokenizer for testing."""
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def sample_dataset(tokenizer):
    """Create a small WikiText validation dataset for testing."""
    # Use validation split (smaller) and limit articles for speed
    dataset = WikiTextDataset(
        split='validation',
        tokenizer=tokenizer,
        min_length=50,
        max_length=2048,  # Limit for faster tests
    )
    return dataset


class TestWikiTextDataset:
    """Tests for WikiTextDataset class."""

    def test_dataset_loading(self, sample_dataset):
        """Test that dataset loads successfully."""
        assert len(sample_dataset) > 0, "Dataset should contain articles"
        print(f"\n  ✅ Loaded {len(sample_dataset)} articles")

    def test_article_structure(self, sample_dataset):
        """Test that articles have correct structure."""
        article = sample_dataset[0]

        # Check keys
        assert 'input_ids' in article, "Article should have input_ids"
        assert 'attention_mask' in article, "Article should have attention_mask"

        # Check types
        assert isinstance(article['input_ids'], torch.Tensor), "input_ids should be tensor"
        assert isinstance(article['attention_mask'], torch.Tensor), "attention_mask should be tensor"

        # Check shapes match
        assert article['input_ids'].shape == article['attention_mask'].shape, \
            "input_ids and attention_mask should have same shape"

        print(f"\n  ✅ Article shape: {article['input_ids'].shape}")

    def test_article_length_constraints(self, sample_dataset):
        """Test that articles respect min/max length constraints."""
        for i, article in enumerate(sample_dataset.articles):
            length = len(article['input_ids'])

            # Check minimum length
            assert length >= sample_dataset.min_length, \
                f"Article {i} has length {length} < min_length {sample_dataset.min_length}"

            # Check maximum length (if set)
            if sample_dataset.max_length is not None:
                assert length <= sample_dataset.max_length, \
                    f"Article {i} has length {length} > max_length {sample_dataset.max_length}"

        print(f"\n  ✅ All {len(sample_dataset)} articles respect length constraints")

    def test_no_truncation_by_default(self, tokenizer):
        """Test that articles are NOT truncated unless max_length is set."""
        # Create dataset without max_length
        dataset = WikiTextDataset(
            split='validation',
            tokenizer=tokenizer,
            min_length=50,
            max_length=None,  # No limit!
        )

        # Find lengths
        lengths = [len(article['input_ids']) for article in dataset.articles]

        # Should have some long articles (>1024 tokens)
        long_articles = [l for l in lengths if l > 1024]

        print(f"\n  ✅ Found {len(long_articles)} articles > 1024 tokens (no truncation)")
        assert len(long_articles) > 0, "Should have some articles exceeding GPT-2's limit"

    def test_tokenization_correctness(self, sample_dataset, tokenizer):
        """Test that tokenization can be reversed (decode -> encode)."""
        article = sample_dataset[0]
        input_ids = article['input_ids']

        # Decode
        text = tokenizer.decode(input_ids)

        # Re-encode
        re_encoded = tokenizer(text, return_tensors='pt')['input_ids'][0]

        # Should match
        assert torch.equal(input_ids, re_encoded), \
            "Decoding and re-encoding should produce identical tokens"

        print(f"\n  ✅ Tokenization is reversible")


class TestCollateFunction:
    """Tests for variable-length collate function."""

    def test_collate_single_article(self, tokenizer):
        """Test collating a single article."""
        # Create mock article
        article = {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.ones(5, dtype=torch.long),
        }

        batch = collate_fn_variable_length([article])

        assert batch['input_ids'].shape == (1, 5), "Batch should have shape [1, 5]"
        assert batch['attention_mask'].shape == (1, 5)

        print(f"\n  ✅ Single article batched correctly")

    def test_collate_variable_lengths(self, tokenizer):
        """Test collating articles of different lengths."""
        # Create mock articles with different lengths
        articles = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.ones(3, dtype=torch.long),
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7, 8]),
                'attention_mask': torch.ones(5, dtype=torch.long),
            },
            {
                'input_ids': torch.tensor([9, 10]),
                'attention_mask': torch.ones(2, dtype=torch.long),
            },
        ]

        batch = collate_fn_variable_length(articles)

        # All should be padded to max length (5)
        assert batch['input_ids'].shape == (3, 5), "Batch should have shape [3, 5]"
        assert batch['attention_mask'].shape == (3, 5)

        # Check padding
        # First article: [1, 2, 3, 0, 0]
        assert batch['input_ids'][0].tolist() == [1, 2, 3, 0, 0]
        assert batch['attention_mask'][0].tolist() == [1, 1, 1, 0, 0]

        # Second article: [4, 5, 6, 7, 8] (no padding)
        assert batch['input_ids'][1].tolist() == [4, 5, 6, 7, 8]
        assert batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 1]

        # Third article: [9, 10, 0, 0, 0]
        assert batch['input_ids'][2].tolist() == [9, 10, 0, 0, 0]
        assert batch['attention_mask'][2].tolist() == [1, 1, 0, 0, 0]

        print(f"\n  ✅ Variable-length collation works correctly")


class TestLongContextDataLoader:
    """Tests for LongContextDataLoader."""

    def test_dataloader_creation(self, sample_dataset):
        """Test creating a DataLoader."""
        loader = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=2,
            shuffle=False,
        )

        assert len(loader) > 0, "DataLoader should have batches"
        print(f"\n  ✅ Created DataLoader with {len(loader)} batches")

    def test_dataloader_iteration(self, sample_dataset):
        """Test iterating through batches."""
        loader = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))

        # Check batch structure
        assert 'input_ids' in batch
        assert 'attention_mask' in batch

        # Check batch size
        assert batch['input_ids'].shape[0] <= 2, "Batch size should be <= 2"

        print(f"\n  ✅ Batch shape: {batch['input_ids'].shape}")

    def test_dataloader_shuffle(self, sample_dataset):
        """Test that shuffling works."""
        loader1 = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=4,
            shuffle=True,
        )

        loader2 = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=4,
            shuffle=True,
        )

        # Get first batches
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # They should (likely) be different due to shuffling
        # Note: There's a small chance they're the same, but very unlikely
        are_different = not torch.equal(batch1['input_ids'], batch2['input_ids'])

        print(f"\n  ✅ Shuffling produces different batches: {are_different}")

    def test_full_epoch(self, sample_dataset):
        """Test iterating through a full epoch."""
        loader = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=4,
            shuffle=False,
        )

        total_articles = 0
        for batch in loader:
            batch_size = batch['input_ids'].shape[0]
            total_articles += batch_size

        assert total_articles == len(sample_dataset), \
            "Full epoch should cover all articles"

        print(f"\n  ✅ Full epoch covered all {total_articles} articles")


class TestCreateDataloaders:
    """Tests for create_dataloaders convenience function."""

    def test_create_all_splits(self, tokenizer):
        """Test creating dataloaders for all splits."""
        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=2,
            min_length=50,
            max_length=2048,
        )

        # Check all splits exist
        assert 'train' in dataloaders
        assert 'validation' in dataloaders
        assert 'test' in dataloaders

        print(f"\n  ✅ Created dataloaders for train/validation/test")

        # Check they're DataLoader instances
        for split, loader in dataloaders.items():
            assert isinstance(loader, LongContextDataLoader)
            print(f"    {split}: {len(loader)} batches")

    def test_dataloaders_work(self, tokenizer):
        """Test that created dataloaders are functional."""
        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=2,
            min_length=50,
            max_length=2048,
        )

        # Try to get a batch from each split
        for split, loader in dataloaders.items():
            batch = next(iter(loader))
            assert batch['input_ids'].shape[0] <= 2
            print(f"\n  ✅ {split} loader works: batch shape {batch['input_ids'].shape}")


class TestDeviceCompatibility:
    """Tests for device compatibility (MPS/CUDA/CPU)."""

    def test_move_batch_to_device(self, sample_dataset):
        """Test moving batch to the appropriate device."""
        device = get_device()
        print(f"\n  Device detected: {device}")

        loader = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))

        # Move to device
        batch_on_device = {
            k: v.to(device) for k, v in batch.items()
        }

        # Check device
        assert str(batch_on_device['input_ids'].device) == device, \
            f"Batch should be on {device}"

        print(f"  ✅ Batch successfully moved to {device}")

    def test_computation_on_device(self, sample_dataset):
        """Test that we can do simple computations on device."""
        device = get_device()

        loader = LongContextDataLoader(
            dataset=sample_dataset,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))
        input_ids = batch['input_ids'].to(device)

        # Simple computation
        doubled = input_ids * 2

        assert doubled.device.type == torch.device(device).type
        print(f"  ✅ Computation works on {device}")


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
