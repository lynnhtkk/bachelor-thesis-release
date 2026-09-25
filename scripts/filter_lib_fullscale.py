"""Full-scale Step 2a threshold + filtering mechanics.

Replicates the per-anchor threshold logic and sentence-building rule from the
reference filter notebooks (``02a_filter_pct_dropoff.ipynb`` and
``02a_filter_kneedle.ipynb``), parameterised so all three values of a method
derive from one already-computed ranking and one filtering pass produces every
configuration at once. Only the scope (which values) is new; the maths is the
notebooks' maths.

Note: unlike ``walk_lib_fullscale``, which imports its scoring helpers from
``walk_lib``, there is no importable ``threshold_lib`` — the reference ranking
(``anchor_sorted_sims``) lives only inline in the notebooks as a closure over
their module globals. It's reproduced here verbatim in logic, just taking
``embeddings``/``pid_to_idx`` as explicit arguments instead of globals.

The threshold constants below are the notebooks' own defaults.
"""

import numpy as np
from kneed import KneeLocator

# ---- Reference constants (the filter notebooks' own values) ----
# pct_dropoff: how deep into each anchor's FULL ranking the drop search looks.
# Bounds only the search, never the ranking itself (which is never truncated).
SEARCH_WINDOW = 50

# kneedle: the fixed baseline theta used as the fallback when no knee is found
# for an anchor (the notebooks' CONSTANT_THETA_BASELINE).
CONSTANT_THETA_BASELINE = 0.70

# Shared filtering rule: discard sentences shorter than this (both notebooks).
MIN_SENTENCE_LENGTH = 2


def anchor_sorted_sims(embeddings, pid_to_idx, anchor_id):
    """Full descending similarity of the whole catalogue to this anchor.

    Reproduces the filter notebooks' ``anchor_sorted_sims`` exactly, but takes
    ``embeddings`` and ``pid_to_idx`` as arguments instead of module globals.

    Mask the anchor's own row to -inf BEFORE sorting: its self-similarity is 1.0
    and, left in, would look like a guaranteed rank-1 cliff for every anchor. The
    ranking is kept FULL (never truncated) — only the search over it bounds depth.
    ``embeddings`` must already be unit-normalised (so the dot product is cosine),
    exactly as the reference prepares it.
    """
    a = pid_to_idx[anchor_id]
    sims = embeddings @ embeddings[a]      # cosine vs anchor (unit-norm); fresh array
    sims[a] = -np.inf                      # mask self before sorting, not after
    return np.sort(sims)[::-1]             # descending; sorted_sims[0] = rank 1


def pct_dropoff_theta(ranked_sims, drop_pct):
    """Percentage drop-off threshold for one anchor's full ranking (ranks 1-based).

    Byte-faithful to the notebook's ``pct_dropoff_theta`` except ``DROP_PCT`` is a
    parameter (``drop_pct``) so one ranking yields every swept value. Walk the
    first ``SEARCH_WINDOW`` entries; at the first consecutive pair whose relative
    drop >= ``drop_pct``, cut there (that rank is the last kept). If nothing clears
    the drop within the window, keep the whole window (graceful fallback).

    Returns ``(theta, cliff_rank_or_None)``. ``cliff_rank`` is ``None`` on the
    no-cliff fallback (the caller reconstructs the reference's ``n_candidates_kept``:
    ``cliff_rank`` when a cliff is found, else ``min(SEARCH_WINDOW, len(ranked_sims))``).
    """
    window = min(SEARCH_WINDOW, len(ranked_sims))
    for i in range(1, window):                     # rank i in 1 .. window-1
        s_i = float(ranked_sims[i - 1])            # score at rank i
        s_next = float(ranked_sims[i])             # score at rank i+1
        if s_i <= 0:                               # degenerate tail; stop searching
            break
        if (s_i - s_next) / s_i >= drop_pct:
            return s_i, i                          # theta = s_i, cut at rank i
    # Fallback: no qualifying drop within the window -> keep the full window.
    return float(ranked_sims[window - 1]), None


def kneedle_theta(ranked_sims, sensitivity, fallback_theta=CONSTANT_THETA_BASELINE):
    """Kneedle threshold for one anchor's full descending ranking.

    Byte-faithful to the notebook's ``kneedle_theta`` except the sensitivity is a
    parameter (``sensitivity`` -> kneed's ``S``) so one ranking yields every swept
    value, and the fallback theta is a parameter (defaulting to the notebooks'
    ``CONSTANT_THETA_BASELINE``). Every other KneeLocator argument matches the
    reference: ``curve="convex"``, ``direction="decreasing"``,
    ``interp_method="interp1d"``.

    x = rank (1-based), y = similarity. ``theta`` is the OBSERVED similarity at the
    detected knee rank (read straight from ``ranked_sims``, not ``kl.knee_y``). No
    knee -> fall back to ``fallback_theta``.

    Returns ``(theta, knee_rank_or_None, fallback_used)``.
    """
    # anchor_sorted_sims parks the masked -inf self value at the tail. Drop that
    # single sentinel before Kneedle: a -inf collapses kneed's [0,1] y-normalisation
    # to NaN (no knee for anyone). What remains is the real catalogue points.
    finite = ranked_sims[np.isfinite(ranked_sims)]
    n = len(finite)
    ranks = np.arange(1, n + 1)   # x = rank, 1-based, same convention as pct_dropoff
    # interp1d, not polynomial: a global polynomial fit over ~49,687 points
    # risks oscillation (Runge's phenomenon) and is unnecessary here.
    kl = KneeLocator(ranks, finite, S=sensitivity,
                     curve="convex", direction="decreasing",
                     interp_method="interp1d")
    if kl.knee is not None:
        knee_rank = int(kl.knee)
        # Observed similarity at that rank, not kl.knee_y — exact since
        # finite[i] == ranked_sims[i] for every top rank.
        theta = float(ranked_sims[knee_rank - 1])
        return theta, knee_rank, False
    return float(fallback_theta), None, True


def filter_walk(w, theta):
    """Rebuild a sentence from a raw walk: anchor + every winner with raw_sim >= theta.

    Verbatim from the reference filter notebooks. ``w`` is one raw-walk record
    (``anchor_id``, ``raw_path``, ``raw_sims``); ``raw_sims[i]`` is the similarity
    of ``raw_path[i + 1]``, so this is a direct zip of ``raw_path[1:]`` against
    ``raw_sims``.
    """
    sentence = [w["anchor_id"]]
    for nxt, sim in zip(w["raw_path"][1:], w["raw_sims"]):
        if sim >= theta:
            sentence.append(nxt)
    return sentence
