"""Build and cache shard index for PrecomputedShardDataset.

Run once before training to avoid scanning shards at training startup.
Cache is saved next to the shard directory as .shard_index.npz.

Usage:
    python scripts/build_shard_index.py \
        --shard-dir data/fbank_shards/wenetspeech_ms
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=str,
                        default="data/fbank_shards/wenetspeech_ms")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    cache_path = shard_dir / ".shard_index.npz"

    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        data = np.load(str(cache_path))
        print(f"Utterances: {len(data['shard_idxs'])}")
        return

    # Fallback: if shard_dir is read-only, save cache one level up
    if not os.access(str(shard_dir), os.W_OK):
        cache_path = shard_dir.parent / f".{shard_dir.name}_shard_index.npz"
        print(f"Note: shard_dir is read-only, cache will be saved to {cache_path}")

    meta_path = shard_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    num_shards = meta["num_shards"]
    print(f"Scanning {num_shards} shards in {shard_dir} ...")

    index_shard = []
    index_within = []
    durations = []

    for shard_idx in tqdm(range(num_shards), desc="Building index",
                          unit="shard", dynamic_ncols=True):
        shard_path = shard_dir / f"shard_{shard_idx:05d}.pt"
        if not shard_path.exists():
            tqdm.write(f"WARNING: missing shard {shard_path}")
            continue
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        n = len(data["features"])
        for i in range(n):
            index_shard.append(shard_idx)
            index_within.append(i)
            durations.append(float(data["durations"][i]))

    np.savez(
        str(cache_path),
        shard_idxs=np.array(index_shard, dtype=np.int32),
        within_idxs=np.array(index_within, dtype=np.int32),
        durations=np.array(durations, dtype=np.float32),
    )
    print(f"\nDone! {len(index_shard)} utterances indexed.")
    print(f"Cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
