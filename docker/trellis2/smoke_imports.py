"""Verifica que las seis extensiones CUDA y el paquete principal importen."""

import importlib
import sys

MODULOS = [
    "torch",
    "trellis2",
    "o_voxel",
    "flash_attn",
    "nvdiffrast",
    "nvdiffrec",
    "cumesh",
    "flexgemm",
]


def main() -> int:
    fallidos = []
    for nombre in MODULOS:
        try:
            importlib.import_module(nombre)
            print(f"  ok    {nombre}")
        except Exception as exc:
            fallidos.append((nombre, f"{type(exc).__name__}: {exc}"))
            print(f"  FALLA {nombre}: {type(exc).__name__}: {exc}")

    import torch

    print(f"\ntorch {torch.__version__} · CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"dispositivo: {torch.cuda.get_device_name(0)}")

    if fallidos:
        print(f"\n{len(fallidos)} módulo(s) no importan:")
        for nombre, error in fallidos:
            print(f"  - {nombre}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
