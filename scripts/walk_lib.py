"""Shared walk mechanics for Step 2 (anchor-fixed semantic walk with basket hopping).

Single source of truth for the walk loop, so the notebook
(``notebooks/02_semantic_walk.ipynb``) and the cluster runner
(``run_walk_cluster.py``) import the exact same behaviour and can never quietly
drift apart.

Behaviour is identical to the walk as it existed inline in the notebook: no
``theta`` / sentence filtering (that was removed from the walk in an earlier
change), records ``raw_path`` and ``raw_sims`` only.
"""

import numpy as np


def apply_score_transform(sims, kind):
    if kind == "identity":
        return sims
    if kind == "squared":
        return sims ** 2
    if kind == "min_max_in_basket":
        lo, hi = float(sims.min()), float(sims.max())
        if hi - lo < 1e-12:
            return np.zeros_like(sims)   # all equal -> uniform softmax
        return (sims - lo) / (hi - lo)
    raise ValueError(f"unknown score_transform: {kind!r}")


def softmax(x, temp):
    z = (x / temp)
    z = z - z.max()                 # numerical stability
    e = np.exp(z)
    w = e / e.sum()
    return w / w.sum()              # guard against float drift for np.choice


def sim_to_anchor(embeddings, pid_to_idx, anchor_idx, cand_ids):
    """Cosine similarity of each candidate to the anchor (embeddings are unit-norm)."""
    rows = [pid_to_idx[c] for c in cand_ids]
    return embeddings[rows] @ embeddings[anchor_idx]


def run_walk(anchor_id, rng, pid_to_idx, embeddings, product_to_orders,
             order_to_products, L, r, tau, score_transform, max_consecutive_empty):
    """One anchor-fixed walk from ``anchor_id``.

    Returns ``{anchor_id, raw_path, raw_sims, hops_taken}``, exactly as the
    notebook's inline version did. ``raw_sims[i]`` is the winning candidate's raw
    cosine similarity to the anchor for ``raw_path[i + 1]`` (appended together).

    ``r`` (walks per starting product) is part of the shared signature so both
    callers pass the run's full parameter set; it is not used within a single
    walk — the caller runs ``r`` walks per anchor.
    """
    anchor_idx = pid_to_idx[anchor_id]
    raw_path = [anchor_id]        # every winner, unfiltered (the anchor, then each hop's winner)
    raw_sims = []                 # winner's raw cosine sim to the anchor; raw_sims[i] -> raw_path[i+1]
    bridge = anchor_id
    visited_baskets = set()
    consecutive_empty = 0
    hops_taken = 0

    for _ in range(L):
        available = product_to_orders.get(bridge, set()) - visited_baskets
        if not available:
            break                                   # no unvisited basket (spec §6.1b)

        # Uniformly sample a basket. sorted() -> canonical order for reproducibility.
        avail_arr = np.array(sorted(available))
        basket = int(rng.choice(avail_arr))
        visited_baskets.add(basket)
        hops_taken += 1

        # Candidates exclude the anchor and current bridge (spec §6.1d, §8).
        candidates = [
            p for p in order_to_products[basket]
            if p != bridge and p != anchor_id and p in pid_to_idx
        ]

        if not candidates:                          # empty basket (spec §6.1e)
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                break
            continue                                # retry from same bridge next hop

        consecutive_empty = 0
        sims = sim_to_anchor(embeddings, pid_to_idx, anchor_idx, candidates)   # raw cosine vs anchor
        weights = softmax(apply_score_transform(sims, score_transform), tau)
        winner_pos = int(rng.choice(len(candidates), p=weights))
        winner = candidates[winner_pos]

        raw_path.append(winner)                               # record the bridge, unfiltered
        raw_sims.append(float(sims[winner_pos]))              # ... and its raw sim to the anchor
        bridge = winner                                       # always move forward (§6.1i)

    return {
        "anchor_id": anchor_id,
        "raw_path": raw_path,
        "raw_sims": raw_sims,
        "hops_taken": hops_taken,
    }
