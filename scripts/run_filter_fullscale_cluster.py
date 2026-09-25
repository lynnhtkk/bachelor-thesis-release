#!/usr/bin/env python
"""Full-scale parallel cluster runner for the Step 2a threshold filtering.

Turns the full-scale walk output (``outputs/full_scale/walk_raw_paths.jsonl``)
into training sentences for all six threshold configurations at once — the
two adaptive methods across their three swept values each:

* ``pct_dropoff`` at ``drop_pct`` = 0.03, 0.05, 0.10
* ``kneedle`` at ``sensitivity`` = 0.3, 0.5, 1.0

The threshold maths and sentence-building rule are the reference notebooks'
(``02a_filter_pct_dropoff.ipynb`` / ``02a_filter_kneedle.ipynb``), reused via
``filter_lib_fullscale``; only the scope (six values in one job, full scale)
is new.

Structure mirrors ``run_walk_fullscale_cluster.py``: fork-based pool, worker
count from ``SLURM_CPUS_PER_TASK``, fixed ``N_CHUNKS`` division, ordered
``imap`` merge, phase-timed prints. Meant to run unattended via ``sbatch``.

Determinism: nothing here is stochastic — ranking, thresholds and filtering
are all deterministic functions of the walk output and embeddings, so a
rerun is bit-for-bit identical.
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

import filter_lib_fullscale
# Chunking logic doesn't care what the chunked work is — reuse it verbatim.
from run_walk_cluster import make_chunks

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FULL_SCALE_DIR = OUT_DIR / "full_scale"
FILTERS_OUT = FULL_SCALE_DIR / "filters"

# ---- The six configurations, all in one place ----
# value_folder follows the existing project convention (confirmed against the
# outputs/skipgram/<method>/<value_folder>/ directories already on disk):
# pct_dropoff -> "drop_{value:.2f}", kneedle -> "sensitivity_{value:.1f}".
# ``param`` is the parameter's key name in the reference's own filter_config.json
# ("drop_pct" for pct_dropoff, "knee_sensitivity" for kneedle).
CONFIGS = [
    {"method": "pct_dropoff", "value": 0.03, "param": "drop_pct",
     "value_folder": "drop_0.03", "label": "pct_dropoff (0.03)"},
    {"method": "pct_dropoff", "value": 0.05, "param": "drop_pct",
     "value_folder": "drop_0.05", "label": "pct_dropoff (0.05)"},
    {"method": "pct_dropoff", "value": 0.10, "param": "drop_pct",
     "value_folder": "drop_0.10", "label": "pct_dropoff (0.10)"},
    {"method": "kneedle", "value": 0.3, "param": "knee_sensitivity",
     "value_folder": "sensitivity_0.3", "label": "kneedle (0.3)"},
    {"method": "kneedle", "value": 0.5, "param": "knee_sensitivity",
     "value_folder": "sensitivity_0.5", "label": "kneedle (0.5)"},
    {"method": "kneedle", "value": 1.0, "param": "knee_sensitivity",
     "value_folder": "sensitivity_1.0", "label": "kneedle (1.0)"},
]

# FIXED chunk count (same default as the walk). Only how many chunks run
# concurrently ever changes, never how the work is divided.
N_CHUNKS = 64

# ---- Globals populated in the main process, inherited by workers via fork ----
EMBEDDINGS = None       # unit-normalised Step 1 embeddings (cosine == dot)
PID_TO_IDX = None       # product_id -> embedding row
CHUNKS = None           # fixed contiguous slices of the walked-anchor list


def load_embeddings():
    """Phase (a): load the existing full-catalog Step 1 embeddings + id map.

    The same files the walk uses (``outputs/embeddings.npy`` /
    ``outputs/product_id_to_index.json``); unit-normalised here so the dot
    product is cosine, exactly as the reference notebooks prepare it.
    """
    t0 = time.perf_counter()
    embeddings = np.load(OUT_DIR / "embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    with open(OUT_DIR / "product_id_to_index.json") as f:
        pid_to_idx = {int(pid): row for pid, row in json.load(f).items()}
    return embeddings, pid_to_idx, time.perf_counter() - t0


def load_and_group_walks():
    """Phase (b): load the full-scale raw walks and group them by anchor_id.

    Reads ``outputs/full_scale/walk_raw_paths.jsonl`` (a large file — ~476MB,
    ~993k lines — hence its own timed phase) and returns:
      * ``walks_by_anchor``: anchor_id -> list of raw-walk records, the per-anchor
        walks the filtering pass in phase (d) needs;
      * ``n_raw_walks``: total walks read;
      * ``walk_config`` + ``walk_config_path``: the walk run this filter is built
        on, for the traceability fields in each filter_config.json.
    """
    t0 = time.perf_counter()
    walks_by_anchor = defaultdict(list)
    n_raw_walks = 0
    with open(FULL_SCALE_DIR / "walk_raw_paths.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = json.loads(line)
            walks_by_anchor[int(w["anchor_id"])].append(w)
            n_raw_walks += 1
    walks_by_anchor = dict(walks_by_anchor)

    walk_config_path = FULL_SCALE_DIR / "walk_config.json"
    walk_config = {}
    if walk_config_path.exists():
        with open(walk_config_path) as f:
            walk_config = json.load(f)

    return walks_by_anchor, n_raw_walks, walk_config, walk_config_path, time.perf_counter() - t0


def process_chunk(chunk_idx):
    """Phase (c) worker: derive all six thresholds for every anchor in one chunk.

    Compute each anchor's ranking ONCE (the expensive matvec + sort), then derive
    every configuration's threshold from that single ranking via
    ``filter_lib_fullscale``. Returns ``(chunk_idx, {anchor_id: {label: entry}})``
    where each ``entry`` is the reference's own thresholds.json schema for that
    method (theta + cliff_rank/knee_rank + n_candidates_kept [+ fallback]).
    """
    results = {}
    for anchor_id in CHUNKS[chunk_idx]:
        if anchor_id not in PID_TO_IDX:
            continue                          # not in the Step 1 catalogue -> cannot rank
        ranked = filter_lib_fullscale.anchor_sorted_sims(EMBEDDINGS, PID_TO_IDX, anchor_id)
        entries = {}
        for cfg in CONFIGS:
            if cfg["method"] == "pct_dropoff":
                theta, cliff_rank = filter_lib_fullscale.pct_dropoff_theta(ranked, cfg["value"])
                # Reconstruct the reference's n_candidates_kept: the cliff rank when
                # a cliff was found, else the full window (its fallback definition).
                n_kept = (cliff_rank if cliff_rank is not None
                          else min(filter_lib_fullscale.SEARCH_WINDOW, len(ranked)))
                entries[cfg["label"]] = {
                    "theta": theta,
                    "cliff_rank": cliff_rank,
                    "n_candidates_kept": int(n_kept),
                }
            else:  # kneedle
                theta, knee_rank, fallback = filter_lib_fullscale.kneedle_theta(ranked, cfg["value"])
                entries[cfg["label"]] = {
                    "theta": theta,
                    "knee_rank": knee_rank,
                    "n_candidates_kept": int(np.sum(ranked >= theta)),   # reference's own definition
                    "fallback": fallback,
                }
        results[anchor_id] = entries
    return chunk_idx, results


def main():
    global EMBEDDINGS, PID_TO_IDX, CHUNKS

    t_total0 = time.perf_counter()

    FILTERS_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64, flush=True)
    print("Full-scale Step 2a filter runner (six configurations)", flush=True)
    print(f"  output root: {FILTERS_OUT}", flush=True)
    print("=" * 64, flush=True)

    # ---- Phase (a): embeddings + id map ----
    print("Phase (a): loading embeddings + product_id_to_index...", flush=True)
    EMBEDDINGS, PID_TO_IDX, phase_a_secs = load_embeddings()
    print(f"  embeddings           : {EMBEDDINGS.shape}", flush=True)
    print(f"  products in id map    : {len(PID_TO_IDX):,}", flush=True)
    print(f"  phase (a) load time   : {phase_a_secs:.2f}s", flush=True)

    # ---- Phase (b): load + group the raw walks ----
    print("Phase (b): loading + grouping full-scale walk_raw_paths.jsonl...", flush=True)
    (walks_by_anchor, n_raw_walks, walk_config,
     walk_config_path, phase_b_secs) = load_and_group_walks()
    walked_anchor_ids = sorted(walks_by_anchor.keys())
    print(f"  raw walks read        : {n_raw_walks:,}", flush=True)
    print(f"  distinct walked anchors: {len(walked_anchor_ids):,}", flush=True)
    print(f"  source walk_config ts  : {walk_config.get('timestamp')}", flush=True)
    print(f"  phase (b) load time    : {phase_b_secs:.2f}s", flush=True)

    # ---- Phase (c): parallel per-anchor thresholds (all six configs) ----
    # Chunk only the anchors that were actually walked (not the full walkable
    # product list) — only those need a threshold.
    CHUNKS = make_chunks(walked_anchor_ids, N_CHUNKS)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)
    print(f"Phase (c): thresholding {len(walked_anchor_ids):,} anchors in "
          f"{N_CHUNKS} chunks across {n_workers} worker(s)...", flush=True)

    thresholds_by_chunk = [None] * N_CHUNKS
    n_done = 0
    t_c0 = time.perf_counter()
    # Linux default 'fork' start method (do NOT set 'spawn'): workers inherit the
    # big globals via copy-on-write. imap preserves order -> deterministic merge.
    with mp.Pool(processes=n_workers) as pool:
        for chunk_idx, results in pool.imap(process_chunk, range(N_CHUNKS)):
            thresholds_by_chunk[chunk_idx] = results
            n_done += 1
            print(f"  chunk {chunk_idx:2d} done: {len(results):,} anchors "
                  f"({n_done}/{N_CHUNKS} chunks)", flush=True)
    phase_c_secs = time.perf_counter() - t_c0

    # Merge in fixed chunk order (deterministic regardless of finish order).
    per_anchor = {}
    for chunk_idx in range(N_CHUNKS):
        per_anchor.update(thresholds_by_chunk[chunk_idx])
    print(f"  phase (c) threshold time: {phase_c_secs:.2f}s "
          f"({len(per_anchor):,} anchors thresholded)", flush=True)

    # ---- Phase (d): one filtering pass, all six configs together ----
    print("Phase (d): single filtering pass across all six configs...", flush=True)
    t_d0 = time.perf_counter()
    kept_by_label = {cfg["label"]: [] for cfg in CONFIGS}
    n_walks_processed = 0
    n_walks_no_threshold = 0
    for anchor_id in walked_anchor_ids:                 # grouped, deterministic order
        entries = per_anchor.get(anchor_id)
        if entries is None:                             # anchor had no computable threshold
            n_walks_no_threshold += len(walks_by_anchor[anchor_id])
            continue
        for w in walks_by_anchor[anchor_id]:
            n_walks_processed += 1
            for cfg in CONFIGS:
                theta = entries[cfg["label"]]["theta"]
                sentence = filter_lib_fullscale.filter_walk(w, theta)
                if len(sentence) >= filter_lib_fullscale.MIN_SENTENCE_LENGTH:
                    kept_by_label[cfg["label"]].append(
                        {"anchor_id": anchor_id, "sentence": sentence})
    phase_d_secs = time.perf_counter() - t_d0
    print(f"  walks processed       : {n_walks_processed:,}", flush=True)
    if n_walks_no_threshold:
        print(f"  walks skipped (no threshold): {n_walks_no_threshold:,}", flush=True)
    print(f"  phase (d) filter time  : {phase_d_secs:.2f}s", flush=True)

    # Timing snapshot recorded into every config's filter_config.json.
    timing = {
        "phase_a_load_embeddings_secs": round(phase_a_secs, 3),
        "phase_b_load_group_walks_secs": round(phase_b_secs, 3),
        "phase_c_parallel_thresholds_secs": round(phase_c_secs, 3),
        "phase_d_filter_pass_secs": round(phase_d_secs, 3),
        "elapsed_through_phase_d_secs": round(time.perf_counter() - t_total0, 3),
    }
    n_anchors = len(per_anchor)
    run_ts = datetime.datetime.now().isoformat(timespec="seconds")

    # ---- Write each config's output as soon as its sentence list is complete ----
    # (one config fully before the next; not all six deferred to a final combined
    # write). Also collect the summary rows as we go.
    summary_rows = []
    for cfg in CONFIGS:
        label, method, value = cfg["label"], cfg["method"], cfg["value"]
        out_dir = FILTERS_OUT / method / cfg["value_folder"]
        out_dir.mkdir(parents=True, exist_ok=True)      # parents=True, exist_ok=True

        kept = kept_by_label[label]
        thresholds = {str(aid): per_anchor[aid][label] for aid in walked_anchor_ids
                      if aid in per_anchor}

        # 1) kept sentences (reference schema: {"anchor_id", "sentence"} per line)
        with open(out_dir / "walk_sentences.jsonl", "w") as f:
            for s in kept:
                f.write(json.dumps(s) + "\n")

        # 2) per-anchor thresholds (reference schema per method)
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f)

        # 3) filter config (+ provenance back to the walk run, + this run's timing)
        n_kept = len(kept)
        n_discarded = n_walks_processed - n_kept
        discard_rate = n_discarded / n_walks_processed if n_walks_processed else 0.0
        filter_config = {
            "method": method,
            cfg["param"]: value,                        # "drop_pct" / "knee_sensitivity"
            "min_sentence_length": filter_lib_fullscale.MIN_SENTENCE_LENGTH,
            "n_raw_walks": n_raw_walks,
            "n_anchors_thresholded": n_anchors,
            "n_sentences_kept": n_kept,
            "discard_rate": round(discard_rate, 6),
            "source_walk_config_timestamp": walk_config.get("timestamp"),
            "source_walk_config_path": str(walk_config_path),
            "timing": timing,
            "timestamp": run_ts,
        }
        if method == "pct_dropoff":
            filter_config["search_window"] = filter_lib_fullscale.SEARCH_WINDOW
        else:  # kneedle: record the knee/fallback split, as the reference does
            n_fallback = sum(1 for aid in per_anchor if per_anchor[aid][label]["fallback"])
            filter_config["n_knee_found"] = n_anchors - n_fallback
            filter_config["n_fallback"] = n_fallback
        with open(out_dir / "filter_config.json", "w") as f:
            json.dump(filter_config, f, indent=2)

        # ---- Summary stats for this config ----
        thetas = np.array([per_anchor[aid][label]["theta"] for aid in per_anchor])
        if method == "pct_dropoff":
            n_fb = sum(1 for aid in per_anchor if per_anchor[aid][label]["cliff_rank"] is None)
        else:
            n_fb = sum(1 for aid in per_anchor if per_anchor[aid][label]["fallback"])
        fallback_rate = n_fb / n_anchors if n_anchors else 0.0
        summary_rows.append({
            "method": method, "value": value,
            "n_anchors": n_anchors, "n_kept": n_kept,
            "discard_rate": discard_rate,
            "mean_theta": float(thetas.mean()) if len(thetas) else 0.0,
            "fallback_rate": fallback_rate,
        })
        print(f"  wrote {label:<20} -> {out_dir}  "
              f"(kept {n_kept:,}, discard {100 * discard_rate:.1f}%)", flush=True)

    total_secs = time.perf_counter() - t_total0

    # ---- Final summary ----
    print("\n" + "=" * 88, flush=True)
    print("SUMMARY — six configurations", flush=True)
    print("=" * 88, flush=True)
    print(f"{'method':<14}{'value':>7}{'n_anchors':>11}{'n_kept':>11}"
          f"{'discard':>10}{'mean_theta':>12}{'fallback':>10}", flush=True)
    print("-" * 88, flush=True)
    for r in summary_rows:
        print(f"{r['method']:<14}{r['value']:>7.2f}{r['n_anchors']:>11,}{r['n_kept']:>11,}"
              f"{100 * r['discard_rate']:>9.1f}%{r['mean_theta']:>12.3f}"
              f"{100 * r['fallback_rate']:>9.1f}%", flush=True)

    print("\nTiming breakdown:", flush=True)
    print(f"  (a) load embeddings    : {timing['phase_a_load_embeddings_secs']:.2f}s", flush=True)
    print(f"  (b) load + group walks : {timing['phase_b_load_group_walks_secs']:.2f}s", flush=True)
    print(f"  (c) parallel thresholds: {timing['phase_c_parallel_thresholds_secs']:.2f}s", flush=True)
    print(f"  (d) filter pass        : {timing['phase_d_filter_pass_secs']:.2f}s", flush=True)
    print(f"  total wall clock       : {total_secs:.2f}s", flush=True)
    print("\nNo randomness anywhere in this job (ranking, thresholds and filtering "
          "are deterministic functions of the walk output) — a rerun is bit-for-bit "
          "identical, so no reproducibility rerun is needed for this stage.", flush=True)


if __name__ == "__main__":
    main()
