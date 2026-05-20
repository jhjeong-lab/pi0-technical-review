from __future__ import annotations

import platform

try:
    import torch
except ImportError:
    torch = None


print("python:", platform.python_version())
print("platform:", platform.platform())

if torch is None:
    print("torch: not installed")
else:
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
    print("gpu_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"gpu[{i}]: {props.name} ({vram_gb:.1f} GB)")
