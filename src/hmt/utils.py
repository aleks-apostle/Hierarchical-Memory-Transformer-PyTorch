"""
Utility functions for HMT
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = None) -> torch.device:
    """
    Get the appropriate device for computation with Apple Silicon (MPS) support.

    Args:
        device: Optional device string ('mps', 'cuda', 'cpu'). If None, auto-detect.

    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        return torch.device(device)

    # Auto-detect device with MPS priority for Apple Silicon
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA backend")
        return torch.device("cuda")
    else:
        logger.info("Using CPU backend")
        return torch.device("cpu")


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    return torch.backends.mps.is_available()


def print_device_info():
    """Print information about available devices."""
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print("Default device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print("Default device: CUDA")
    else:
        print("Default device: CPU")
    print("=" * 60)


def test_device_performance(device: torch.device = None, matrix_size: int = 1024):
    """
    Test device performance with a simple matrix multiplication.

    Args:
        device: Device to test. If None, use auto-detected device.
        matrix_size: Size of square matrices to multiply.
    """
    import time

    if device is None:
        device = get_device()

    print(f"\nTesting {device} performance with {matrix_size}x{matrix_size} matrices...")

    # Create random matrices
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Warmup
    _ = torch.matmul(a, b)

    # Benchmark
    start_time = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    if device.type == "mps":
        torch.mps.synchronize()  # Wait for MPS operations to complete
    elif device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average time per matmul: {avg_time * 1000:.2f} ms")
    print(f"TFLOPS: {(2 * matrix_size**3) / (avg_time * 1e12):.2f}")

    return avg_time
