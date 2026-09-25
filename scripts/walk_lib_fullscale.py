"""Full-scale walk mechanics for Step 2 (anchor-fixed semantic walk).

Behaviourally identical to ``walk_lib.run_walk`` — same candidate exclusion,
softmax scoring, and empty-hop termination. The only change is how a random
not-yet-visited basket is picked at each hop, for performance only. Scoring /
transform / similarity helpers are imported from ``walk_lib``, not
reimplemented, so they stay byte-identical.

Why: ``walk_lib.run_walk`` rebuilds and sorts the full unvisited-basket set at
every hop. For a popular product that set can hold tens of thousands of
basket IDs, just to avoid re-visiting at most ``L = 15`` of them. This module
removes that repeated gather-and-sort.

Determinism: since the sampling mechanism changed, walks won't match
``walk_lib.run_walk`` bit-for-bit even with the same seed. What must hold
(checked by ``validate_walk_lib_fullscale.py``) is that each hop still draws
uniformly at random from the bridge product's unvisited baskets.
"""

import numpy as np

from walk_lib import apply_score_transform, softmax, sim_to_anchor

# Above this many total baskets, use the fast rejection-sampling path.
# visited_baskets never exceeds L = 15 for a whole walk, so any product with
# more baskets than that is guaranteed an unvisited one; 50 sits comfortably
# above that floor.
FAST_PATH_MIN_BASKETS = 50

# Cap on rejection-sampling redraws on the fast path. Should never be hit
# given the size guarantee above; falls back to the exact method if it is.
MAX_REJECTION_ATTEMPTS = 20


def build_basket_arrays(product_to_orders):
    """Precompute a sorted int32 numpy array of basket IDs per product.

    Takes the reference's set-based ``product_to_orders`` and returns
    ``dict[int, np.ndarray]``. Sorting matches the reference's ``sorted()``
    for reproducibility, not because the per-hop sampling needs it sorted.
    Runs once instead of per hop, which is the point.
    """
    return {
        int(pid): np.array(sorted(oids), dtype=np.int32)
        for pid, oids in product_to_orders.items()
    }


def run_walk(anchor_id, rng, pid_to_idx, embeddings, product_basket_arrays,
             order_to_products, L, r, tau, score_transform, max_consecutive_empty):
    """One anchor-fixed walk from ``anchor_id`` — full-scale basket selection.

    Same signature and return schema as ``walk_lib.run_walk``, except it
    takes ``product_basket_arrays`` (from :func:`build_basket_arrays`)
    instead of the set-based ``product_to_orders``. ``r`` is part of the
    shared signature but unused within a single walk. Only basket selection
    differs from ``walk_lib.run_walk``; everything downstream is identical.
    """
    anchor_idx = pid_to_idx[anchor_id]
    raw_path = [anchor_id]        # every winner, unfiltered (the anchor, then each hop's winner)
    raw_sims = []                 # winner's raw cosine sim to the anchor; raw_sims[i] -> raw_path[i+1]
    bridge = anchor_id
    visited_baskets = set()
    consecutive_empty = 0
    hops_taken = 0

    for _ in range(L):
        arr = product_basket_arrays.get(bridge)
        if arr is None or len(arr) == 0:
            break                                   # no available basket (matches reference §6.1b)

        # Basket selection: the only change from walk_lib.run_walk.
        if len(arr) > FAST_PATH_MIN_BASKETS:
            # Rejection sampling: draw a uniform index, redraw if visited.
            # Exact uniformity over unvisited baskets, not an approximation;
            # rejections are rare given the FAST_PATH_MIN_BASKETS guarantee.
            basket = None
            for _attempt in range(MAX_REJECTION_ATTEMPTS):
                cand = int(arr[rng.integers(0, len(arr))])
                if cand not in visited_baskets:
                    basket = cand
                    break
            if basket is None:
                # Shouldn't happen given the size guarantee; exact fallback.
                available = set(arr.tolist()) - visited_baskets
                if not available:
                    break
                basket = int(rng.choice(np.array(sorted(available))))
        else:
            # Few baskets: cheap to use the reference's exact method directly.
            available = set(arr.tolist()) - visited_baskets
            if not available:
                break                               # no unvisited basket (reference §6.1b)
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
