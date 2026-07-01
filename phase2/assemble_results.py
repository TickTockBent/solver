"""Assemble Phase 2 (depth x data 2x2) results into markdown tables.

Reads runs/phase2/<cell>_s<seed>/{heldout.json, train_summary.json}, averages p over
seeds per cell, and prints: (1) per-cell held-out quality, (2) per-run compute, and
(3) the compounding interaction. Handles partial results -- run it any time as cells
land. Paste the output into phase2/phase2-results.md.

    python phase2/assemble_results.py --runs lnhm/runs/phase2 [--mode sampled|greedy]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

CELLS = [("C00_control", "C00 control"), ("C10_depth", "C10 depth"),
         ("C01_data", "C01 data"), ("C11_both", "C11 both")]


def load_runs(runs_dir):
    """-> {cell: {seed: {'p': {level: p}, 'compute': {...}}}}"""
    data = defaultdict(dict)
    for cell_id, _label in CELLS:
        for run_dir in sorted(glob.glob(os.path.join(runs_dir, f"{cell_id}_s*"))):
            seed = run_dir.rsplit("_s", 1)[-1]
            entry = {}
            heldout_path = os.path.join(run_dir, "heldout.json")
            if os.path.exists(heldout_path):
                with open(heldout_path) as f:
                    entry["levels"] = json.load(f).get("levels", {})
            summary_path = os.path.join(run_dir, "train_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    entry["compute"] = json.load(f)
            if entry:
                data[cell_id][seed] = entry
    return data


def cell_mean_p(data, cell_id, mode):
    """Mean p per level over seeds for a cell. -> {level_int: mean_p}"""
    key = f"p_{mode}"
    per_level = defaultdict(list)
    for _seed, entry in data.get(cell_id, {}).items():
        for level_str, result in entry.get("levels", {}).items():
            if key in result:
                per_level[int(level_str)].append(result[key])
    return {lvl: sum(v) / len(v) for lvl, v in per_level.items()}


def fmt(x):
    return f"{x:.3f}" if x is not None else "_"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--mode", choices=["sampled", "greedy"], default="sampled")
    args = parser.parse_args()

    data = load_runs(args.runs)
    means = {cell_id: cell_mean_p(data, cell_id, args.mode) for cell_id, _ in CELLS}
    levels = sorted({lvl for m in means.values() for lvl in m})
    if not levels:
        print(f"No heldout results found under {args.runs} yet.")
        return 0

    header = "| cell | " + " | ".join(f"n{l}" for l in levels) + " |"
    sep = "|" + "---|" * (len(levels) + 1)
    print(f"## Held-out quality (p, {args.mode}), mean over seeds\n\n{header}\n{sep}")
    for cell_id, label in CELLS:
        row = " | ".join(fmt(means[cell_id].get(l)) for l in levels)
        print(f"| {label} | {row} |")

    print("\n## Compute per run\n")
    print("| cell | seed | total steps | instances seen | wall (s) |")
    print("|---|---|---|---|---|")
    for cell_id, label in CELLS:
        for seed, entry in sorted(data.get(cell_id, {}).items()):
            c = entry.get("compute")
            if c:
                print(f"| {label} | {seed} | {c['total_steps']} | {c['instances_seen']} | {c['wall_seconds']} |")

    print(f"\n## Interaction (Delta p vs control, {args.mode})\n")
    print("| n | Δ_depth | Δ_data | Δ_both | interaction |")
    print("|---|---|---|---|---|")
    control, depth, data_, both = (means["C00_control"], means["C10_depth"],
                                   means["C01_data"], means["C11_both"])
    for lvl in levels:
        if lvl not in control:
            continue
        d_depth = depth.get(lvl, None)
        d_data = data_.get(lvl, None)
        d_both = both.get(lvl, None)
        base = control[lvl]
        dd = (d_depth - base) if d_depth is not None else None
        da = (d_data - base) if d_data is not None else None
        db = (d_both - base) if d_both is not None else None
        inter = (db - dd - da) if None not in (dd, da, db) else None
        cells = " | ".join(f"{x:+.3f}" if x is not None else "_" for x in (dd, da, db, inter))
        print(f"| {lvl} | {cells} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
