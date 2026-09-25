#!/usr/bin/env python
"""Full-scale Step 3 Skip-gram (SGNS) training, one model per configuration.

Trains six Word2Vec models — one per threshold configuration — from the
full-scale filtered corpora produced by ``run_filter_fullscale_cluster.py``,
writing each model's artifacts under
``outputs/full_scale/skipgram/{method}/{value_folder}/``.

Hyperparameters and corpus/save mechanics are the reference notebook's
(``03_skipgram.ipynb``), reused unchanged; only the scope (six configs at
full scale, driven from files) is new.

Parallelism, deliberately different from the walk/filter jobs: those fan out
with a multiprocessing Pool and pin BLAS threads to 1 to stop forked workers
oversubscribing cores. gensim's Word2Vec is already internally multithreaded
via ``workers``, so here the six models train sequentially, each given the
full core count — no BLAS thread-pinning block, and none should be added.

Worker count comes from ``SLURM_CPUS_PER_TASK`` (falling back to
``os.cpu_count()``), not hardcoded, since student-VM nodes have 56 cores.

Determinism: gensim's Word2Vec is only bit-for-bit reproducible with
``workers=1``; with more workers, OS thread scheduling makes update order
nondeterministic even with a fixed seed. This job takes throughput over
exact reproducibility — seed and worker count are recorded in each model's
``train_config.json`` along with a note that multi-worker runs don't
reproduce exactly.

Meant to run unattended via ``sbatch``, not interactively.
"""

import os
import json
import time
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FULL_SCALE_DIR = OUT_DIR / "full_scale"
FILTERS_DIR = FULL_SCALE_DIR / "filters"
SKIPGRAM_DIR = FULL_SCALE_DIR / "skipgram"

# ---- Word2Vec / SGNS hyperparameters — IDENTICAL to notebooks/03_skipgram.ipynb ----
# (Every value below is copied from the reference EXCEPT the worker count: the
# notebook trains with workers=4, whereas this job hands each model the full core
# count resolved at runtime — see n_workers in main(). Nothing else differs.)
SG = 1                # 1 = Skip-gram (not CBOW), per Item2Vec
VECTOR_SIZE = 128     # embedding dimensionality
WINDOW = 10           # >= max sentence length -> every in-sentence pair is a positive
MIN_COUNT = 1         # CRITICAL: keep every product; gensim default 5 drops sparse anchors
NEGATIVE = 15         # negative samples per positive
SAMPLE = 1e-3         # frequent-token subsampling
EPOCHS = 20
SEED = 42             # recorded, but see the reproducibility note (multi-worker != reproducible)

# Force retraining of every config even if complete artifacts already exist.
# Left False so a killed run resumes cheaply (only missing configs are trained).
RETRAIN_ALL = False

# The six configurations. Replicated verbatim from run_filter_fullscale_cluster.py
# (same structure + same value_folder names, matching the on-disk folders) rather
# than imported: importing that module would execute its top-level BLAS-pinning
# env sets (OMP_NUM_THREADS=1 ...), exactly the pinning this job must avoid.
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

REPRODUCIBILITY_NOTE = (
    "Multi-worker Word2Vec training is NOT bit-for-bit reproducible across runs: "
    "with workers > 1, OS thread scheduling makes the order of weight updates "
    "nondeterministic even with a fixed seed. The seed and worker count are "
    "recorded to document the run; exact reproduction requires workers=1."
)

# The complete artifact set that marks a config as already trained (resume check).
ARTIFACT_FILES = ("model.gensim", "embeddings.npy", "product_id_to_index.json",
                  "train_config.json")


def build_corpus(sentences_path):
    """Load a configuration's kept sentences into gensim's expected form.

    Byte-faithful to 03_skipgram.ipynb's ``build_corpus``: one sentence per walk
    record (duplicates and order kept as-is), each product ID converted to a
    string token with ``str(pid)``.
    """
    records = []
    with open(sentences_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [[str(pid) for pid in rec["sentence"]] for rec in records]


def is_complete(out_dir):
    """True iff every artifact of a finished training run is already present."""
    return all((out_dir / fn).exists() for fn in ARTIFACT_FILES)


def train_config(cfg, n_workers):
    """Train, save, and summarise one configuration. Returns a summary-row dict.

    Writes this config's artifacts as soon as training finishes (before the caller
    moves on to the next config), so a late failure never costs earlier successes.
    """
    method, value, value_folder = cfg["method"], cfg["value"], cfg["value_folder"]
    in_path = FILTERS_DIR / method / value_folder / "walk_sentences.jsonl"
    out_dir = SKIPGRAM_DIR / method / value_folder

    corpus = build_corpus(in_path)
    lengths = [len(s) for s in corpus]
    token_counts = Counter(tok for s in corpus for tok in s)
    total_tokens = sum(lengths)
    print(f"  sentences: {len(corpus):,}  (distinct tokens {len(token_counts):,}, "
          f"total tokens {total_tokens:,})", flush=True)

    # ---- Train (identical hyperparameters to the reference; workers = full grant) ----
    t0 = time.perf_counter()
    model = Word2Vec(
        sentences=corpus,
        sg=SG,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        negative=NEGATIVE,
        sample=SAMPLE,
        epochs=EPOCHS,
        workers=n_workers,
        seed=SEED,
    )
    train_secs = time.perf_counter() - t0
    vocab_size = len(model.wv)
    print(f"  vocab size: {vocab_size:,}  train time: {train_secs:.2f}s", flush=True)

    # ---- Save artifacts: same files + same save calls as 03_skipgram.ipynb ----
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = list(model.wv.index_to_key)
    sg_embeddings = model.wv.vectors                    # (vocab, VECTOR_SIZE), raw (unnormalised)
    sg_pid_to_idx = {int(tok): i for i, tok in enumerate(tokens)}

    model.save(str(out_dir / "model.gensim"))
    np.save(out_dir / "embeddings.npy", sg_embeddings)
    with open(out_dir / "product_id_to_index.json", "w") as f:
        json.dump({str(pid): i for pid, i in sg_pid_to_idx.items()}, f)

    # ---- Traceability: the filter run this corpus came from ----
    filter_cfg_path = FILTERS_DIR / method / value_folder / "filter_config.json"
    filter_cfg_timestamp = None
    if filter_cfg_path.exists():
        with open(filter_cfg_path) as f:
            filter_cfg_timestamp = json.load(f).get("timestamp")

    # ---- train_config.json (superset of the notebook's config.json) ----
    train_config_doc = {
        "method": method,
        "param": cfg["param"],
        "value": value,
        # Every Word2Vec hyperparameter actually used:
        "sg": SG,
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "negative": NEGATIVE,
        "sample": SAMPLE,
        "epochs": EPOCHS,
        "workers": n_workers,
        "seed": SEED,
        # Corpus + model stats:
        "n_sentences": len(corpus),
        "n_distinct_tokens": len(token_counts),
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        # Timing + reproducibility + provenance:
        "train_seconds": round(train_secs, 3),
        "reproducibility_note": REPRODUCIBILITY_NOTE,
        "source_filter_config_path": str(filter_cfg_path),
        "source_filter_config_timestamp": filter_cfg_timestamp,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(train_config_doc, f, indent=2)

    print(f"  wrote -> {out_dir}", flush=True)
    return {"method": method, "value": value, "n_sentences": len(corpus),
            "vocab_size": vocab_size, "train_seconds": train_secs, "reused": False}


def summary_row_from_existing(cfg):
    """Build a summary row for an already-complete (skipped) config from its saved
    train_config.json, so the final table still shows every config."""
    out_dir = SKIPGRAM_DIR / cfg["method"] / cfg["value_folder"]
    doc = {}
    try:
        with open(out_dir / "train_config.json") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {"method": cfg["method"], "value": cfg["value"],
            "n_sentences": doc.get("n_sentences"),
            "vocab_size": doc.get("vocab_size"),
            "train_seconds": doc.get("train_seconds"),
            "reused": True}


def main():
    t_total0 = time.perf_counter()
    SKIPGRAM_DIR.mkdir(parents=True, exist_ok=True)

    # Worker count from Slurm's grant, falling back to os.cpu_count() locally.
    # NOT hardcoded (studvm nodes have 56 cores, not 64). gensim uses these as
    # intra-model training threads; models are trained one at a time.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)

    print("=" * 72, flush=True)
    print("Full-scale Step 3 Skip-gram training (six configurations, sequential)", flush=True)
    print(f"  workers per model : {n_workers}", flush=True)
    print(f"  seed              : {SEED}  (multi-worker => not bit-for-bit reproducible)", flush=True)
    print(f"  RETRAIN_ALL       : {RETRAIN_ALL}", flush=True)
    print(f"  output root       : {SKIPGRAM_DIR}", flush=True)
    print("=" * 72, flush=True)

    summary_rows = []
    for i, cfg in enumerate(CONFIGS, start=1):
        out_dir = SKIPGRAM_DIR / cfg["method"] / cfg["value_folder"]
        print(f"\n[{i}/{len(CONFIGS)}] {cfg['label']}", flush=True)

        if not RETRAIN_ALL and is_complete(out_dir):
            print(f"  already complete -> skipping (set RETRAIN_ALL=True to force)", flush=True)
            summary_rows.append(summary_row_from_existing(cfg))
            continue

        summary_rows.append(train_config(cfg, n_workers))

    total_secs = time.perf_counter() - t_total0

    # ---- Final summary table ----
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY — six configurations", flush=True)
    print("=" * 72, flush=True)
    print(f"{'method':<14}{'value':>7}{'n_sentences':>14}{'vocab_size':>13}"
          f"{'train_s':>11}   note", flush=True)
    print("-" * 72, flush=True)
    for r in summary_rows:
        ns = f"{r['n_sentences']:,}" if r['n_sentences'] is not None else "?"
        vs = f"{r['vocab_size']:,}" if r['vocab_size'] is not None else "?"
        ts = f"{r['train_seconds']:.1f}" if r['train_seconds'] is not None else "?"
        note = "reused (skipped)" if r["reused"] else ""
        print(f"{r['method']:<14}{r['value']:>7.2f}{ns:>14}{vs:>13}{ts:>11}   {note}",
              flush=True)

    print(f"\ntotal wall clock: {total_secs:.2f}s", flush=True)
    print(REPRODUCIBILITY_NOTE, flush=True)


if __name__ == "__main__":
    main()
