import sys

import torch


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 无法使用 CUDA。请检查是否安装了 CUDA 版本的 PyTorch。"
        )

    device = torch.device("cuda:0")

    print("GPU name:", torch.cuda.get_device_name(device))
    print("Compute capability:", torch.cuda.get_device_capability(device))

    properties = torch.cuda.get_device_properties(device)
    print(
        "Total VRAM:",
        f"{properties.total_memory / 1024**3:.2f} GB",
    )

    # 执行一次真正的 GPU 半精度矩阵运算。
    x = torch.randn(
        (4096, 4096),
        device=device,
        dtype=torch.float16,
    )
    y = x @ x

    torch.cuda.synchronize()

    print("Tensor device:", y.device)
    print("Tensor dtype:", y.dtype)
    print("Result shape:", tuple(y.shape))
    print(
        "Allocated VRAM:",
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
    )
    print("GPU test passed.")


if __name__ == "__main__":
    main()