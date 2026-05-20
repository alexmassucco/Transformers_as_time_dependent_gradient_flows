"""
Main CLI for running attention gradient-flow experiments.
"""

import argparse
import importlib
from typing import Dict


EXPERIMENTS: Dict[str, str] = {
    "s1_oscillating": "experiments.s1_oscillating",
    "s1_ou": "experiments.s1_ou",
    "s2_oscillating": "experiments.s2_oscillating",
    "s2_ou": "experiments.s2_ou",
    "s2_combined_plots": "experiments.s2_combined_plots",
    "hd_oscillating": "experiments.hd_oscillating",
    "hd_ou": "experiments.hd_ou",
    "hd_combined_plots": "experiments.hd_combined_plots",
}


def _load_experiment(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Use `list` to see options.")
    module = importlib.import_module(EXPERIMENTS[name])
    return module


def _list_experiments():
    print("Available experiments:")
    for name, modpath in EXPERIMENTS.items():
        module = importlib.import_module(modpath)
        desc = getattr(module, "DESCRIPTION", "").strip()
        print(f"  {name:18s}  {desc}")


def _build_parser():
    parser = argparse.ArgumentParser(description="Attention gradient-flow experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available experiments")

    run_p = sub.add_parser("run", help="Run one or more experiments")
    run_p.add_argument("name", nargs="+", help="Experiment name(s) or 'all'")
    run_p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    run_p.add_argument("--gif", action="store_true", help="Create GIFs")
    run_p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    run_p.add_argument("--T", type=float, help="Final time")
    run_p.add_argument("--dt", type=float, help="Time step")
    run_p.add_argument("--n", type=int, help="Number of tokens")
    run_p.add_argument("--H", type=int, nargs="+", help="Head counts")
    run_p.add_argument("--n-mc", dest="n_mc", type=int, help="Monte Carlo paths")
    run_p.add_argument("--mc-batch", dest="mc_batch", type=int, help="MC batch size")
    run_p.add_argument("--noise-var", dest="noise_var", type=float, help="Noise variance")
    run_p.add_argument("--d", type=int, help="Ambient dimension (high-d experiments)")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed")
    run_p.add_argument("--results-dir", default="./results", help="Results directory")
    run_p.add_argument("--figures-dir", default="./figures", help="Figures directory")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list":
        _list_experiments()
        return

    if args.command == "run":
        names = args.name
        if len(names) == 1 and names[0] == "all":
            names = [k for k in EXPERIMENTS.keys() if not k.endswith("_plots")]
        for name in names:
            module = _load_experiment(name)
            module.run(args)


if __name__ == "__main__":
    main()
