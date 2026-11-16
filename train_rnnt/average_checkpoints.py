"""Average the last N RNN-T checkpoints."""

import argparse
from pathlib import Path

import torch


def average_checkpoints(ckpt_paths, out_path):
    avg_state = None
    n = len(ckpt_paths)
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_state[k] += v.float()
        print(f"  + {p}")

    for k in avg_state:
        avg_state[k] /= n

    out = {
        "model": avg_state,
        "epoch": "avg",
        "averaged_from": [str(p) for p in ckpt_paths],
    }
    torch.save(out, out_path)
    print(f"Saved averaged checkpoint ({n} models) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="Explicit epoch numbers, e.g. 46 47 48 49 50")
    parser.add_argument("--last-n", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if args.epochs:
        paths = [exp_dir / f"epoch_{e}.pt" for e in args.epochs]
    else:
        paths = sorted(exp_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        paths = paths[-args.last_n:]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    out = Path(args.output) if args.output else exp_dir / f"avg_last{len(paths)}.pt"
    print(f"Averaging {len(paths)} checkpoints:")
    average_checkpoints(paths, out)


if __name__ == "__main__":
    main()
