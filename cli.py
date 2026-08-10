"""Punto de entrada: `python cli.py run experiments/smoke.yaml`."""

import argparse
import sys
from pathlib import Path

import backends.local  # noqa: F401  registra el backend
import models.mock  # noqa: F401  registra el modelo
from core.experiment import load_experiment, run_experiment
from core.registry import available_backends, available_models
from core.runstore import RunStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opengames")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Ejecuta un experimento")
    run_cmd.add_argument("config", type=Path)
    run_cmd.add_argument("--runs-dir", type=Path, default=Path("runs"))

    sub.add_parser("list", help="Muestra modelos y backends registrados")

    args = parser.parse_args(argv)

    if args.command == "list":
        print("Modelos: ", ", ".join(available_models()))
        print("Backends:", ", ".join(available_backends()))
        return 0

    results = run_experiment(load_experiment(args.config), RunStore(args.runs_dir))
    for result in results:
        marca = "cache" if result.cached else "nuevo"
        print(f"[{marca}] {result.run_id}")
    print(f"{len(results)} corridas, {sum(r.cached for r in results)} desde caché")
    return 0


if __name__ == "__main__":
    sys.exit(main())
