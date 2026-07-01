"""Wrap a raw state_dict checkpoint into the self-describing format eval_heldout wants.

The early Phase-0 runs (runs/first, runs/second, runs/full) saved a bare
``state_dict``. ``eval_heldout.py`` requires ``{state_dict, model_config}``. This
re-saves a raw checkpoint with an explicit model_config alongside it (non-destructive:
writes a new file). For runs/full the arch is d_model=128 / 3 layers / 8 heads /
ff_dim=512 (inferred from the tensor shapes; 710K params).

    python analysis/wrap_checkpoint.py runs/full/model_final.pt runs/full/model_selfdesc.pt
"""
from __future__ import annotations

import argparse

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Raw state_dict checkpoint.")
    parser.add_argument("dst", help="Output path for the self-describing checkpoint.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    arguments = parser.parse_args()

    state_dict = torch.load(arguments.src, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        print(f"{arguments.src} is already self-describing; copying through.")
        checkpoint = state_dict
    else:
        checkpoint = {
            "state_dict": state_dict,
            "model_config": {
                "d_model": arguments.d_model,
                "n_encoder_layers": arguments.layers,
                "n_heads": arguments.heads,
                "ff_dim": arguments.ff_dim,
            },
        }
    torch.save(checkpoint, arguments.dst)
    print(f"-> {arguments.dst}  (config: {checkpoint['model_config']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
