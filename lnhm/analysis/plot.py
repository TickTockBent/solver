"""Metric visualization for LNHM Phase 0.

The key chart: accuracy by level over training epochs, one line per level — the
chart Phase 0 lives or dies on. Reads the metrics.csv written by training/train.py.

    python analysis/plot.py --metrics runs/phase0/metrics.csv --out runs/phase0/accuracy_by_level.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_metrics(metrics_path: str) -> Dict[int, List[Tuple[int, float]]]:
    """Return {level: [(global_epoch, accuracy), ...]} from a metrics CSV."""
    accuracy_by_level: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    with open(metrics_path, newline="") as metrics_file:
        for row in csv.DictReader(metrics_file):
            level = int(row["level"])
            accuracy_by_level[level].append((int(row["global_epoch"]), float(row["accuracy"])))
    return accuracy_by_level


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Plot accuracy-by-level over epochs.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv from training.")
    parser.add_argument("--out", required=True, help="Output image path (e.g. .png).")
    arguments = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accuracy_by_level = load_metrics(arguments.metrics)
    figure, axes = plt.subplots(figsize=(10, 6))
    for level in sorted(accuracy_by_level):
        epochs, accuracies = zip(*accuracy_by_level[level])
        axes.plot(epochs, accuracies, marker="", label=f"n={level}")

    axes.set_xlabel("training epoch")
    axes.set_ylabel("validation accuracy (within 1% of optimum)")
    axes.set_title("LNHM Phase 0 — accuracy by level over training")
    axes.set_ylim(0, 1)
    axes.grid(True, alpha=0.3)
    axes.legend(title="level", ncol=2)
    figure.tight_layout()
    figure.savefig(arguments.out, dpi=120)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
