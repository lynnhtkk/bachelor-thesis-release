#!/usr/bin/env python
"""Full-scale parallel cluster runner for the Step 2 anchor-fixed semantic walk.

Same walk, same outputs as ``run_walk_cluster.py``, but at full scale: every
order in ``data/order_products__prior.csv``, every product with at least one
basket as a starting anchor, and the faster basket selection from
``walk_lib_fullscale`` (rejection sampling instead of gather-and-sort per
hop). Meant to run unattended via ``sbatch``, not interactively.

The chunking / fork / SeedSequence / imap-ordered-merge model is inherited
verbatim from ``run_walk_cluster.py`` (``make_chunks`` imported from it).
Only the walk call and the one-time basket-array precompute change.

Benchmark mode: optional env vars, all defaulting to real full-scale
behaviour when unset, let a reduced run measure how phases scale before
committing cluster resources to the real run:

* ``FULLSCALE_BENCHMARK_N_STARTING`` — random subset of starting products of
  this size (``sample_seed = 7``) instead of all.
* ``FULLSCALE_BENCHMARK_N_CHUNKS`` — override the chunk count for this run.
* ``FULLSCALE_OUT_DIR`` — override the output directory (default
  ``outputs/full_scale``).

Active overrides are recorded in the run's own output so a benchmark run is
never mistaken for a real one.
"""

# Pin BLAS threading to 1 BEFORE importing numpy, so each worker process stays
# single-threaded and workers don't oversubscribe the cores Slurm granted.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import datetime
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import walk_lib_fullscale
# Chunking logic is unaffected by the basket-selection change — reuse it verbatim.
from run_walk_cluster import make_chunks

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs" / "full_scale"

# ---- Walk mechanics (identical to the reference; unchanged at full scale) ----
L = 15
r = 20
tau = 0.15
score_transform = "identity"
random_seed = 42            # root seed for walk stochasticity
max_consecutive_empty = 3

# FIXED chunk count (same default as the reference). Only how many chunks run
# concurrently ever changes, never how the work is divided.
N_CHUNKS = 64

# Dedicated seed for the benchmark starting-product subsample — kept separate
# from the walk's random_seed (42), the same separation-of-concerns the
# sample-scale notebook already uses.
sample_seed = 7

# ---- Globals populated in the main process, inherited by workers via fork ----
EMBEDDINGS = None
PID_TO_IDX = None
PRODUCT_BASKET_ARRAYS = None      # precomputed sorted arrays (the fast-path input)
ORDER_TO_PRODUCTS = None
CHUNKS = None


def load_inputs():
    """Load Step 1 embeddings and build the FULL basket graph (no sampling).

    Returns embeddings, pid_to_idx, product_to_orders (set-based, as the
    reference builds it), order_to_products, the full sorted walkable
    starting-product list, and the phase-(a) build time in seconds.
    """
    t0 = time.perf_counter()

    embeddings = np.load(ROOT / "outputs" / "embeddings.npy").astype(np.float32)
    # Unit-normalise -> cosine similarity becomes a dot product (same as notebook).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    with open(ROOT / "outputs" / "product_id_to_index.json") as f:
        pid_to_idx = {int(pid): row for pid, row in json.load(f).items()}

    # FULL scale: read the entire prior, no order sampling / subsetting at all.
    prior = pd.read_csv(
        DATA_DIR / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        dtype={"order_id": "int32", "product_id": "int32"},
    )

    # Build the basket graph with the same structure the reference builds
    # (defaultdict(set) for product_to_orders, one call site, unchanged).
    product_to_orders = defaultdict(set)
    order_to_products = defaultdict(list)
    for oid, pid in zip(prior["order_id"].to_numpy(), prior["product_id"].to_numpy()):
        oid, pid = int(oid), int(pid)
        product_to_orders[pid].add(oid)
        order_to_products[oid].append(pid)
    product_to_orders = dict(product_to_orders)
    order_to_products = dict(order_to_products)

    # Every product with at least one basket is a walkable anchor (the complete
    # population). Sorted -> deterministic list, no random draw needed here.
    starting_products = sorted(product_to_orders.keys())

    build_secs = time.perf_counter() - t0
    return (embeddings, pid_to_idx, product_to_orders, order_to_products,
            starting_products, build_secs)


def process_chunk(args):
    """Run ``r`` walks for every starting product in one chunk. Runs in a worker.

    Reads the large shared inputs from module globals (inherited via fork), and
    takes only the small per-chunk arguments (index + its spawned seed). Uses
    the full-scale walk with the precomputed basket arrays.
    """
    chunk_idx, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    results = []
    for anchor_id in CHUNKS[chunk_idx]:
        if anchor_id not in PRODUCT_BASKET_ARRAYS:
            continue                      # no baskets -> cannot walk (matches reference)
        for _ in range(r):
            w = walk_lib_fullscale.run_walk(
                anchor_id, rng, PID_TO_IDX, EMBEDDINGS,
                PRODUCT_BASKET_ARRAYS, ORDER_TO_PRODUCTS,
                L, r, tau, score_transform, max_consecutive_empty,
            )
            results.append(w)
    return chunk_idx, results


def main():
    global EMBEDDINGS, PID_TO_IDX, PRODUCT_BASKET_ARRAYS, ORDER_TO_PRODUCTS, CHUNKS

    t_total0 = time.perf_counter()

    # ---- Resolve benchmark-mode overrides (default to full-scale when unset) ----
    bench_n_starting = os.environ.get("FULLSCALE_BENCHMARK_N_STARTING")
    bench_n_chunks = os.environ.get("FULLSCALE_BENCHMARK_N_CHUNKS")
    out_dir_override = os.environ.get("FULLSCALE_OUT_DIR")

    bench_n_starting = int(bench_n_starting) if bench_n_starting else None
    n_chunks = int(bench_n_chunks) if bench_n_chunks else N_CHUNKS
    out_dir = Path(out_dir_override) if out_dir_override else DEFAULT_OUT_DIR

    # The runner creates its own output directory (parents ok, idempotent).
    out_dir.mkdir(parents=True, exist_ok=True)

    active_overrides = {}
    if bench_n_starting is not None:
        active_overrides["FULLSCALE_BENCHMARK_N_STARTING"] = bench_n_starting
    if bench_n_chunks is not None:
        active_overrides["FULLSCALE_BENCHMARK_N_CHUNKS"] = int(bench_n_chunks)
    if out_dir_override is not None:
        active_overrides["FULLSCALE_OUT_DIR"] = out_dir_override
    benchmark_mode = bool(active_overrides)

    print("=" * 64, flush=True)
    print("Full-scale walk runner "
          f"({'BENCHMARK' if benchmark_mode else 'REAL FULL SCALE'})", flush=True)
    if benchmark_mode:
        print(f"  active overrides: {active_overrides}", flush=True)
    print(f"  output dir: {out_dir}", flush=True)
    print("=" * 64, flush=True)

    # ---- Phase (a): read CSV + build the basket graph ----
    print("Phase (a): reading prior + building basket graph (FULL scale)...", flush=True)
    (EMBEDDINGS, PID_TO_IDX, product_to_orders, ORDER_TO_PRODUCTS,
     starting_products_full, phase_a_secs) = load_inputs()
    n_orders_total = len(ORDER_TO_PRODUCTS)
    print(f"  embeddings            : {EMBEDDINGS.shape}", flush=True)
    print(f"  orders (baskets)      : {n_orders_total:,}", flush=True)
    print(f"  walkable products     : {len(starting_products_full):,}", flush=True)
    print(f"  phase (a) build time  : {phase_a_secs:.2f}s", flush=True)

    # Per-product basket frequency — same schema as the reference, from the same
    # set-based product_to_orders before it is converted to arrays.
    product_basket_freq = {
        "total_baskets": n_orders_total,
        "product_basket_count": {int(pid): len(oids) for pid, oids in product_to_orders.items()},
    }
    with open(out_dir / "product_basket_freq.json", "w") as f:
        json.dump(product_basket_freq, f)
    print(f"Saved product_basket_freq.json "
          f"({len(product_basket_freq['product_basket_count']):,} products, "
          f"{product_basket_freq['total_baskets']:,} baskets)", flush=True)

    # ---- Benchmark: optionally reduce the starting-product set ----
    if bench_n_starting is not None and bench_n_starting < len(starting_products_full):
        rng_sub = np.random.default_rng(sample_seed)
        idx = rng_sub.choice(len(starting_products_full), size=bench_n_starting, replace=False)
        # Sort the drawn subset -> deterministic, stable chunk assignment.
        starting_products = sorted(int(starting_products_full[i]) for i in idx)
        print(f"  BENCHMARK: sampled {len(starting_products):,} of "
              f"{len(starting_products_full):,} starting products "
              f"(sample_seed={sample_seed})", flush=True)
    else:
        starting_products = starting_products_full

    # ---- Phase (b): build the precomputed basket arrays (one-time) ----
    print("Phase (b): building precomputed basket arrays...", flush=True)
    t_b0 = time.perf_counter()
    PRODUCT_BASKET_ARRAYS = walk_lib_fullscale.build_basket_arrays(product_to_orders)
    phase_b_secs = time.perf_counter() - t_b0
    # The set-based version is no longer needed; free it before forking so
    # workers don't inherit a redundant copy via copy-on-write.
    del product_to_orders
    print(f"  phase (b) precompute time: {phase_b_secs:.2f}s "
          f"({len(PRODUCT_BASKET_ARRAYS):,} product arrays)", flush=True)

    # ---- Fixed chunks + one spawned child seed per chunk ----
    CHUNKS = make_chunks(starting_products, n_chunks)
    chunk_seeds = np.random.SeedSequence(random_seed).spawn(n_chunks)

    # Worker count from Slurm's grant, falling back to os.cpu_count() locally.
    # This only sets concurrency; the work division is fixed at n_chunks.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)
    print(f"Phase (c): running {n_chunks} chunks across {n_workers} worker(s)...", flush=True)

    tasks = list(zip(range(n_chunks), chunk_seeds))
    raw_paths_by_chunk = [None] * n_chunks
    n_done = 0
    n_walks = 0

    # ---- Phase (c): the parallel walk ----
    t_c0 = time.perf_counter()
    # Relies on the Linux default 'fork' start method so workers inherit the
    # big globals via copy-on-write. imap preserves chunk order.
    with mp.Pool(processes=n_workers) as pool:
        for chunk_idx, results in pool.imap(process_chunk, tasks):
            raw_paths_by_chunk[chunk_idx] = results
            n_done += 1
            n_walks += len(results)
            print(f"  chunk {chunk_idx:2d} done: {len(results):,} walks "
                  f"({n_done}/{n_chunks} chunks, {n_walks:,} walks total)", flush=True)
    phase_c_secs = time.perf_counter() - t_c0

    # Merge in fixed chunk order (deterministic regardless of finish order).
    raw_paths = []
    for chunk_idx in range(n_chunks):
        raw_paths.extend(raw_paths_by_chunk[chunk_idx])

    total_secs = time.perf_counter() - t_total0
    timing = {
        "phase_a_read_and_build_graph_secs": round(phase_a_secs, 3),
        "phase_b_build_basket_arrays_secs": round(phase_b_secs, 3),
        "phase_c_parallel_walk_secs": round(phase_c_secs, 3),
        "phase_d_total_wall_clock_secs": round(total_secs, 3),
    }

    # ---- Save outputs: same filenames + schema as the reference ----
    with open(out_dir / "walk_raw_paths.jsonl", "w") as f:
        for w in raw_paths:
            f.write(json.dumps(w) + "\n")

    config = {
        "L": L,
        "r": r,
        "tau": tau,
        "score_transform": score_transform,
        "random_seed": random_seed,
        "sample_seed": sample_seed if bench_n_starting is not None else None,
        "n_orders_sample": None,          # no order sampling at full scale
        "n_starting_products": len(starting_products),
        "max_consecutive_empty": max_consecutive_empty,
        "n_walks": len(raw_paths),
        # ---- Cluster-execution provenance ----
        "execution": "parallel_cluster_fullscale",
        "n_chunks": n_chunks,
        "n_workers": n_workers,
        "root_seed": random_seed,
        # ---- Basket-selection implementation provenance ----
        "basket_selection_impl": "walk_lib_fullscale",
        "fast_path_min_baskets": walk_lib_fullscale.FAST_PATH_MIN_BASKETS,
        "max_rejection_attempts": walk_lib_fullscale.MAX_REJECTION_ATTEMPTS,
        # ---- Benchmark provenance ----
        "benchmark_mode": benchmark_mode,
        "benchmark_overrides": active_overrides,
        # ---- Four-phase timing breakdown ----
        "timing": timing,
        "reproducibility_note": (
            "Walk order and raw_path contents are reproducible across reruns of "
            "this script with the same root_seed and n_chunks. They are NOT "
            "expected to match run_walk_cluster.py: the basket-selection "
            "mechanism (rejection sampling) changed the random-draw sequence, "
            "though each hop still samples uniformly over unvisited baskets."
        ),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "walk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # shared_sample.json: filename kept for downstream compatibility, contents
    # honestly reflect what actually happened. The full order-id list is
    # deliberately omitted (it would be enormous and pointless at full scale);
    # n_orders_used records the true total instead.
    shared_sample = {
        "sampling_applied": bench_n_starting is not None,
        "n_orders_used": n_orders_total,
        "all_orders_used": True,          # every basket in the prior was used
        "sampled_order_ids": None,        # not subsetted; omitted by design
        "starting_products": starting_products,
        "n_starting_products": len(starting_products),
        "sample_seed": sample_seed if bench_n_starting is not None else None,
        "benchmark_mode": benchmark_mode,
        "benchmark_overrides": active_overrides,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "shared_sample.json", "w") as f:
        json.dump(shared_sample, f)

    print(f"Saved walk_raw_paths.jsonl ({len(raw_paths):,} walks), "
          f"walk_config.json, product_basket_freq.json, shared_sample.json "
          f"-> {out_dir}", flush=True)
    print("Timing breakdown:", flush=True)
    print(f"  (a) read + build graph : {timing['phase_a_read_and_build_graph_secs']:.2f}s", flush=True)
    print(f"  (b) build basket arrays: {timing['phase_b_build_basket_arrays_secs']:.2f}s", flush=True)
    print(f"  (c) parallel walk      : {timing['phase_c_parallel_walk_secs']:.2f}s", flush=True)
    print(f"  (d) total wall clock   : {timing['phase_d_total_wall_clock_secs']:.2f}s", flush=True)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
