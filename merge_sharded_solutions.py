#!/usr/bin/env python3
"""
Merge Hugging Face dataset shards saved via `save_to_disk` with names like:
  <prefix>_shard_K_of_N

Example:
  deepscaler_baseline_solutions_shard_0_of_11

Usage:
  python merge_hf_shards.py --dir . --prefix deepscaler_baseline_solutions --out deepscaler_baseline_solutions_merged --delete
"""

import argparse
import os
import re
import shutil
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict


def find_shards(root: str, prefix: str):
    rx = re.compile(rf"^{re.escape(prefix)}_shard_(\d+)_of_(\d+)$")
    shards = []
    for name in os.listdir(root):
        m = rx.match(name)
        if not m:
            continue
        k = int(m.group(1))
        n = int(m.group(2))
        shards.append((k, n, name))
    shards.sort(key=lambda x: x[0])
    if not shards:
        raise SystemExit(f"No shards found in {root} matching {prefix}_shard_K_of_N")

    # sanity: all report same N
    Ns = {n for _, n, _ in shards}
    if len(Ns) != 1:
        raise SystemExit(f"Shards disagree on N: {sorted(Ns)}")

    # sanity: indices look contiguous (optional but useful)
    n = shards[0][1]
    ks = [k for k, _, _ in shards]
    if ks != list(range(len(ks))) or (n != len(ks) and max(ks) + 1 != n):
        # don’t hard-fail; just warn via exception message if you want strictness
        pass

    return [(k, n, os.path.join(root, name)) for k, n, name in shards]


def merge_datasets(ds_list):
    first = ds_list[0]
    if isinstance(first, Dataset):
        return concatenate_datasets(ds_list)

    if isinstance(first, DatasetDict):
        splits = list(first.keys())
        for ds in ds_list[1:]:
            if not isinstance(ds, DatasetDict) or list(ds.keys()) != splits:
                raise SystemExit("Shard types/splits mismatch (Dataset vs DatasetDict or split names differ).")
        merged = DatasetDict()
        for split in splits:
            merged[split] = concatenate_datasets([d[split] for d in ds_list])
        return merged

    raise SystemExit(f"Unexpected object from load_from_disk: {type(first)}")


def delete_paths(paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory containing shard folders")
    ap.add_argument("--prefix", required=True, help="Prefix before _shard_K_of_N")
    ap.add_argument("--out", required=True, help="Output dataset directory (save_to_disk target)")
    ap.add_argument("--delete", action="store_true", help="Delete shard folders after successful merge+save")
    ap.add_argument("--dry-run", action="store_true", help="List shards and exit")
    args = ap.parse_args()

    shard_infos = find_shards(args.dir, args.prefix)
    shard_paths = [p for _, _, p in shard_infos]

    print("Found shards:")
    for k, n, p in shard_infos:
        print(f"  shard {k} of {n}: {p}")

    if args.dry_run:
        return

    print("\nLoading shards...")
    ds_list = [load_from_disk(p) for p in shard_paths]

    print("Merging...")
    merged = merge_datasets(ds_list)

    out_path = os.path.join(args.dir, args.out) if not os.path.isabs(args.out) else args.out
    if os.path.exists(out_path):
        raise SystemExit(f"Output path already exists: {out_path}")

    print(f"Saving merged dataset to: {out_path}")
    merged.save_to_disk(out_path)

    # basic verification
    if isinstance(merged, Dataset):
        print(f"Merged rows: {len(merged)}")
    else:
        for split, d in merged.items():
            print(f"Merged split '{split}' rows: {len(d)}")

    if args.delete:
        print("Deleting shard directories...")
        delete_paths(shard_paths)
        print("Done.")


if __name__ == "__main__":
    main()
