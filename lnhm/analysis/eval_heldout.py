"""Held-out one-shot evaluation for the depth x data 2x2 overtraining experiment.

Loads a trained checkpoint and decodes a FIXED held-out test set (greedy + sampled
best-of-K) per level, reporting the optimality gap and p = 1/(1+mean_gap). The test
set is disjoint from training by construction (generated with a distinct base_seed),
and its optimal-distance labels are baked in (exact Held-Karp for n<=12, LKH beyond),
so evaluation is a pure decode-and-compare -- no re-solving.

Usage:
    python analysis/eval_heldout.py --checkpoint runs/<run>/model_final.pt \
        --test-dir data/phase0_test --levels 5 8 10 12 16 20 25 30 \
        --samples 16 --limit 1000 --out runs/<run>/heldout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.dataset import load_level_arrays  # noqa: E402
from model.lnhm import LnhmModel  # noqa: E402
from training.evaluate import tour_distance_torch  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device) -> LnhmModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not (isinstance(checkpoint, dict) and "state_dict" in checkpoint and "model_config" in checkpoint):
        raise ValueError(
            f"{checkpoint_path} is not a self-describing checkpoint (needs state_dict + model_config). "
            "Re-train with the current train.py, which saves model_config."
        )
    model = LnhmModel.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_level(model, coordinates_np, optimal_np, device, samples, batch_size=1024):
    """Greedy + best-of-`samples` sampled decode. Returns per-level gap/p dict."""
    greedy_gaps, sampled_gaps = [], []
    for start in range(0, len(coordinates_np), batch_size):
        coordinates = torch.from_numpy(coordinates_np[start:start + batch_size]).to(device)
        optimal = torch.from_numpy(optimal_np[start:start + batch_size]).to(device)
        current_batch = coordinates.shape[0]

        greedy_tours, _ = model.solve(coordinates, mode="greedy")
        greedy_distance = tour_distance_torch(coordinates, greedy_tours)
        greedy_gaps.append(((greedy_distance - optimal) / optimal.clamp(min=1e-9)).cpu().numpy())

        if samples > 0:
            tiled = coordinates.repeat(samples, 1, 1)  # (samples*B, n, 2)
            sampled_tours, _ = model.solve(tiled, mode="sample")
            sampled_distance = tour_distance_torch(tiled, sampled_tours).view(samples, current_batch)
            best_distance = sampled_distance.min(dim=0).values
            sampled_gaps.append(((best_distance - optimal) / optimal.clamp(min=1e-9)).cpu().numpy())

    greedy = np.concatenate(greedy_gaps)
    result = {
        "n_instances": int(len(greedy)),
        "greedy_mean_gap": float(greedy.mean()),
        "p_greedy": float(1.0 / (1.0 + greedy.mean())),
    }
    if samples > 0:
        sampled = np.concatenate(sampled_gaps)
        result.update({
            "samples": int(samples),
            "sampled_mean_gap": float(sampled.mean()),
            "p_sampled": float(1.0 / (1.0 + sampled.mean())),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True, help="Dir with level_XX_train.jsonl.gz held-out files.")
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=16, help="Best-of-K sampled decode (0 = greedy only).")
    parser.add_argument("--limit", type=int, default=1000, help="Per-level eval instances.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Write per-level JSON here.")
    parser.add_argument("--split", default="train", help="Split filename in test-dir (default: train).")
    arguments = parser.parse_args()

    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(arguments.checkpoint, device)

    results = {}
    print(f"{'n':>4} {'inst':>6} {'greedy_gap%':>12} {'p_greedy':>9} {'sampled_gap%':>13} {'p_sampled':>10}")
    for level in arguments.levels:
        path = os.path.join(arguments.test_dir, f"level_{level:02d}_{arguments.split}.jsonl.gz")
        if not os.path.exists(path):
            print(f"{level:>4}  (missing {path}; skipped)")
            continue
        coordinates, _tours, distances = load_level_arrays(path, arguments.limit)
        result = evaluate_level(model, coordinates, distances, device, arguments.samples)
        results[level] = result
        sampled_gap = result.get("sampled_mean_gap", float("nan")) * 100
        p_sampled = result.get("p_sampled", float("nan"))
        print(f"{level:>4} {result['n_instances']:>6} {result['greedy_mean_gap']*100:>12.2f} "
              f"{result['p_greedy']:>9.3f} {sampled_gap:>13.2f} {p_sampled:>10.3f}")

    if arguments.out:
        with open(arguments.out, "w") as out_file:
            json.dump({"checkpoint": arguments.checkpoint, "levels": results}, out_file, indent=2)
        print(f"-> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
