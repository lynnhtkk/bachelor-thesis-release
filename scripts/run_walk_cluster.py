#!/usr/bin/env python
"""Parallel cluster runner for the Step 2 anchor-fixed semantic walk.

Meant to run unattended via ``sbatch submit_walk.sbatch``, not interactively.
Produces the same outputs (filenames + schema) as ``02_semantic_walk.ipynb``,
parallelised across starting products.

Determinism: the order sample and starting-product list are loaded from
``outputs/shared_sample.json`` rather than re-drawn from seeds, so results
don't depend on numpy's RNG stream matching across versions/platforms. Walk
stochasticity uses one child seed per chunk, spawned from a single
``SeedSequence(root_seed)`` — reproducible across reruns of this script with
the same seed and chunk count, but not expected to match a single-threaded
local run.
"""

# Pin BLAS threading to 1 BEFORE importing numpy, so each worker process stays
# single-threaded and workers don't oversubscribe the cores Slurm granted.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import datetime
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import walk_lib

# ROOT resolves to the same directory the notebooks treat as ROOT, regardless
# of the current working directory.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# ---- Walk mechanics (must match the notebook's parameters) ----
L = 15
r = 20
tau = 0.15
score_transform = "identity"
random_seed = 42            # root seed for walk stochasticity
max_consecutive_empty = 3

# FIXED chunk count. This is what makes a rerun reproducible even if a future
# submission requests a different core count: only how many chunks run
# concurrently changes, never how the work is divided.
N_CHUNKS = 64

# ---- Globals populated in the main process, inherited by workers via fork ----
# Building these once before the pool is created lets Linux's copy-on-write fork
# share them with workers, instead of pickling a separate copy into each one.
EMBEDDINGS = None
PID_TO_IDX = None
PRODUCT_TO_ORDERS = None
ORDER_TO_PRODUCTS = None
CHUNKS = None


def load_inputs():
    """Load Step 1 embeddings + the shared sample, and build the basket graph."""
    embeddings = np.load(OUT_DIR / "embeddings.npy").astype(np.float32)
    # Unit-normalise -> cosine similarity becomes a dot product (same as notebook).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    with open(OUT_DIR / "product_id_to_index.json") as f:
        pid_to_idx = {int(pid): row for pid, row in json.load(f).items()}

    # The canonical shared sample — do NOT redraw from seeds; load it verbatim.
    with open(OUT_DIR / "shared_sample.json") as f:
        shared = json.load(f)
    sampled_order_ids = set(int(x) for x in shared["sampled_order_ids"])
    starting_products = [int(p) for p in shared["starting_products"]]

    # Read the prior once, keep only rows from the shared sample's orders, then
    # build the basket graph with the same structure the notebook builds.
    prior = pd.read_csv(
        DATA_DIR / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        dtype={"order_id": "int32", "product_id": "int32"},
    )
    sample = prior[prior["order_id"].isin(sampled_order_ids)]

    product_to_orders = defaultdict(set)
    order_to_products = defaultdict(list)
    for oid, pid in zip(sample["order_id"].to_numpy(), sample["product_id"].to_numpy()):
        oid, pid = int(oid), int(pid)
        product_to_orders[pid].add(oid)
        order_to_products[oid].append(pid)
    product_to_orders = dict(product_to_orders)
    order_to_products = dict(order_to_products)

    return (embeddings, pid_to_idx, product_to_orders, order_to_products,
            starting_products, shared)


def make_chunks(starting_products, n_chunks):
    """Split starting_products into ``n_chunks`` contiguous slices, in order.

    Contiguous slices in the order they appear in shared_sample.json, so the
    division of work is fixed regardless of the runtime core count.
    """
    idx_chunks = np.array_split(np.arange(len(starting_products)), n_chunks)
    return [[starting_products[j] for j in part.tolist()] for part in idx_chunks]


def process_chunk(args):
    """Run ``r`` walks for every starting product in one chunk. Runs in a worker.

    Reads the large shared inputs from module globals (inherited via fork), and
    takes only the small per-chunk arguments (index + its spawned seed).
    """
    chunk_idx, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    results = []
    for anchor_id in CHUNKS[chunk_idx]:
        if anchor_id not in PRODUCT_TO_ORDERS:
            continue                      # no baskets -> cannot walk (matches notebook)
        for _ in range(r):
            w = walk_lib.run_walk(
                anchor_id, rng, PID_TO_IDX, EMBEDDINGS,
                PRODUCT_TO_ORDERS, ORDER_TO_PRODUCTS,
                L, r, tau, score_transform, max_consecutive_empty,
            )
            results.append(w)
    return chunk_idx, results


def main():
    global EMBEDDINGS, PID_TO_IDX, PRODUCT_TO_ORDERS, ORDER_TO_PRODUCTS, CHUNKS

    print("Loading inputs (embeddings, shared sample, basket graph)...", flush=True)
    (EMBEDDINGS, PID_TO_IDX, PRODUCT_TO_ORDERS, ORDER_TO_PRODUCTS,
     starting_products, shared) = load_inputs()
    print(f"  embeddings        : {EMBEDDINGS.shape}", flush=True)
    print(f"  starting products : {len(starting_products):,}", flush=True)
    print(f"  baskets           : {len(ORDER_TO_PRODUCTS):,}", flush=True)

    # Per-product basket frequency — from this same product_to_orders, identical
    # schema to the notebook's output.
    product_basket_freq = {
        "total_baskets": len(ORDER_TO_PRODUCTS),
        "product_basket_count": {int(pid): len(oids) for pid, oids in PRODUCT_TO_ORDERS.items()},
    }
    with open(OUT_DIR / "product_basket_freq.json", "w") as f:
        json.dump(product_basket_freq, f)
    print(f"Saved product_basket_freq.json "
          f"({len(product_basket_freq['product_basket_count']):,} products, "
          f"{product_basket_freq['total_baskets']:,} baskets)", flush=True)

    # Fixed 64 contiguous chunks + one spawned child seed per chunk.
    CHUNKS = make_chunks(starting_products, N_CHUNKS)
    chunk_seeds = np.random.SeedSequence(random_seed).spawn(N_CHUNKS)

    # Worker count comes from Slurm's grant (SLURM_CPUS_PER_TASK), falling back to
    # os.cpu_count() for local off-cluster testing. This only sets concurrency;
    # the work division is fixed at N_CHUNKS.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)
    print(f"Running {N_CHUNKS} chunks across {n_workers} worker(s)...", flush=True)

    tasks = list(zip(range(N_CHUNKS), chunk_seeds))
    raw_paths_by_chunk = [None] * N_CHUNKS
    n_done = 0
    n_walks = 0

    # Relies on the Linux default 'fork' start method (not 'spawn') so workers
    # inherit the big globals via copy-on-write. imap preserves chunk order,
    # so we can stream progress and merge deterministically.
    with mp.Pool(processes=n_workers) as pool:
        for chunk_idx, results in pool.imap(process_chunk, tasks):
            raw_paths_by_chunk[chunk_idx] = results
            n_done += 1
            n_walks += len(results)
            print(f"  chunk {chunk_idx:2d} done: {len(results):,} walks "
                  f"({n_done}/{N_CHUNKS} chunks, {n_walks:,} walks total)", flush=True)

    # Merge in fixed chunk order 0..63 (deterministic regardless of finish order).
    raw_paths = []
    for chunk_idx in range(N_CHUNKS):
        raw_paths.extend(raw_paths_by_chunk[chunk_idx])

    # ---- Save outputs: same filenames + schema as the notebook ----
    with open(OUT_DIR / "walk_raw_paths.jsonl", "w") as f:
        for w in raw_paths:
            f.write(json.dumps(w) + "\n")

    config = {
        "L": L,
        "r": r,
        "tau": tau,
        "score_transform": score_transform,
        "random_seed": random_seed,
        # Sample parameters come from the shared sample, not re-derived here.
        "sample_seed": shared.get("sample_seed"),
        "n_orders_sample": shared.get("n_orders_sample"),
        "n_starting_products": len(starting_products),
        "max_consecutive_empty": max_consecutive_empty,
        "n_walks": len(raw_paths),
        # ---- Cluster-execution provenance ----
        "execution": "parallel_cluster",
        "n_chunks": N_CHUNKS,
        "n_workers": n_workers,
        "root_seed": random_seed,
        "reproducibility_note": (
            "Walk order and raw_path contents are reproducible across reruns of "
            "this script with the same root_seed and n_chunks, but are NOT "
            "expected to match a single-threaded local run bit-for-bit."
        ),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_DIR / "walk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved walk_raw_paths.jsonl ({len(raw_paths):,} walks), "
          f"walk_config.json, product_basket_freq.json", flush=True)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
