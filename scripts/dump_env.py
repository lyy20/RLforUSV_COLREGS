"""训练环境快照：本地校验过的依赖版本，用于在租用服务器上复现同一环境。

用法：python scripts/dump_env.py
输出可直接用于 `pip install -r` 或对照服务器环境。
"""
import importlib
import importlib.metadata as md
import platform
import sys

# 代码中实际 import 的第三方包 -> PyPI 包名
PACKAGES = [
    ("torch", "torch"), ("numpy", "numpy"), ("gym", "gym"),
    ("matplotlib", "matplotlib"), ("tensorboardX", "tensorboardX"),
    ("imageio", "imageio"), ("pyglet", "pyglet"), ("PIL", "Pillow"),
    ("cv2", "opencv-python"), ("bitarray", "bitarray"),
    ("cloudpickle", "cloudpickle"), ("six", "six"), ("tqdm", "tqdm"),
    ("requests", "requests"), ("yaml", "PyYAML"), ("scipy", "scipy"),
    ("tensorboard", "tensorboard"), ("matplotlib.pyplot", "matplotlib"),
]


def main() -> None:
    print("# 训练环境快照 (python scripts/dump_env.py)")
    print("python           =", sys.version.split()[0])
    print("platform         =", platform.platform())
    print("processor        =", platform.processor())
    print()
    print("## 关键依赖版本")
    for mod_name, dist_name in PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None) or md.version(dist_name)
            print(f"{dist_name:20s} = {ver}")
        except Exception as exc:  # noqa: BLE001 - 快照脚本要容忍缺失
            print(f"{dist_name:20s} = MISSING ({type(exc).__name__})")
    print()
    print("## CUDA")
    try:
        import torch
        print("torch.version.cuda =", torch.version.cuda)
        print("cuda.is_available  =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device_count       =", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {prop.name} {prop.total_memory // (1024 ** 2)} MiB")
    except Exception as exc:  # noqa: BLE001
        print("CUDA 查询失败:", exc)


if __name__ == "__main__":
    main()
