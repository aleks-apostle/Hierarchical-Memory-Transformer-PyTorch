"""
Test script to verify Apple Silicon MPS acceleration is working.
"""

import sys
sys.path.insert(0, "src")

import torch
from hmt.utils import (
    get_device,
    is_mps_available,
    print_device_info,
    test_device_performance,
)


def test_mps_basic():
    """Test basic MPS functionality."""
    print("\n=== Testing MPS Basic Functionality ===\n")

    device = get_device()
    print(f"Selected device: {device}")

    # Test tensor creation
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)

    # Test basic operations
    z = x + y
    w = torch.matmul(x, y)

    print(f"✓ Tensor creation successful on {device}")
    print(f"✓ Addition successful: output shape {z.shape}")
    print(f"✓ Matrix multiplication successful: output shape {w.shape}")

    # Test moving tensors
    cpu_tensor = w.cpu()
    device_tensor = cpu_tensor.to(device)
    print(f"✓ Tensor movement between CPU and {device} successful")

    return True


def test_transformer_compatibility():
    """Test that transformers library works with MPS."""
    print("\n=== Testing Transformers Library Compatibility ===\n")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = get_device()
        print(f"Loading small GPT-2 model on {device}...")

        # Load a small model for testing
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to device
        model = model.to(device)
        print(f"✓ Model loaded and moved to {device}")

        # Test inference
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✓ Inference successful")
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Output logits shape: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"✗ Error testing transformers: {e}")
        return False


def run_all_tests():
    """Run all MPS tests."""
    print("\n" + "=" * 60)
    print("HMT Apple Silicon (MPS) Test Suite")
    print("=" * 60)

    # Print device info
    print_device_info()

    # Check MPS availability
    if not is_mps_available():
        print("\n⚠️  WARNING: MPS is not available on this system!")
        print("The project will fall back to CPU, which will be slower.")
        print("\nTo enable MPS, ensure you have:")
        print("  - macOS 12.3 or later")
        print("  - PyTorch 1.12 or later with MPS support")
        return False

    # Run tests
    results = []

    # Basic MPS test
    try:
        results.append(("Basic MPS Operations", test_mps_basic()))
    except Exception as e:
        print(f"\n✗ Basic MPS test failed: {e}")
        results.append(("Basic MPS Operations", False))

    # Performance test
    try:
        print("\n=== Performance Benchmark ===\n")
        device = get_device()
        test_device_performance(device, matrix_size=1024)
        results.append(("Performance Test", True))
    except Exception as e:
        print(f"\n✗ Performance test failed: {e}")
        results.append(("Performance Test", False))

    # Transformers compatibility test
    try:
        results.append(("Transformers Compatibility", test_transformer_compatibility()))
    except Exception as e:
        print(f"\n✗ Transformers test failed: {e}")
        results.append(("Transformers Compatibility", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! MPS is ready for HMT training.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
