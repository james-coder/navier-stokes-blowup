"""Run `python -m ns_blowup --help` for a ready-to-run simulation."""
import argparse
import json
from pathlib import Path

import jax

from .simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Simulate a periodic Navier–Stokes benchmark")
    parser.add_argument("--resolution", type=int, default=32, help="points per axis (default: 32)")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--flow", choices=["taylor-green", "abc"], default="taylor-green")
    parser.add_argument("--nu", type=float, default=0.01, help="kinematic viscosity")
    parser.add_argument("--time", type=float, default=1.0, help="end time")
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    try:
        sim = Simulation((args.resolution,) * args.dim, nu=args.nu, dtype=args.precision, device=args.device)
        result = sim.run(args.flow, t_end=args.time, frames=args.frames)
        result.save(args.output / "flow.npz")
        result.export_vtk(args.output / "final.vtk")
        diagnostics = {k: v.tolist() for k, v in result.diagnostics().items()}
        (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        if args.plot:
            result.plot(path=args.output / "speed.png")
        print(json.dumps({**result.metadata, "output": str(args.output)}, indent=2))
    except (ValueError, RuntimeError, FloatingPointError, ImportError) as exc:
        parser.exit(2, f"ns-blowup: {exc}\n")


if __name__ == "__main__":
    main()
