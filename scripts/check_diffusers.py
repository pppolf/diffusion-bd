import sys

import accelerate
import diffusers
import safetensors
import torch
import transformers


def main() -> None:
    print("Python executable:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Diffusers:", diffusers.__version__)
    print("Transformers:", transformers.__version__)
    print("Accelerate:", accelerate.__version__)
    print("Safetensors:", safetensors.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Diffusers environment test passed.")


if __name__ == "__main__":
    main()