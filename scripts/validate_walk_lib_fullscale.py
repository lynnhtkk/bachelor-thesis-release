#!/usr/bin/env python
"""Standalone correctness check for the full-scale basket-selection rewrite.

No dataset needed, runs in well under a second. This is what to run locally
before trusting anything on the cluster.

What it proves: the new fast rejection-sampling path in
``walk_lib_fullscale`` draws uniformly at random from a bridge product's
not-yet-visited baskets, exactly like the reference's exact method
(``available = set(arr) - visited_baskets`` then uniform choice). The exact walks
differ (the random-draw sequence changed), but the *per-hop basket sampling
distribution* must be identical: uniform over the same available set.

It does NOT import or run any walk over real data. It isolates and compares the
two basket-picking mechanisms directly on synthetic arrays.
"""

import numpy as np

from walk_lib_fullscale import FAST_PATH_MIN_BASKETS, MAX_REJECTION_ATTEMPTS

# Empirical-uniformity tolerance: max allowed relative deviation of any
# available basket's observed frequency from the expected uniform frequency.
# A genuinely skewed sampler blows past this; honest uniform sampling stays well
# under it PROVIDED the pool is small enough that each basket gets many draws
# (see the per-scenario draw counts below — a large pool with too few draws
# fails this by pure variance even when perfectly uniform).
MAX_REL_DEVIATION = 0.15

# Default draw count. Scenarios 2 and 3 are correctly calibrated here: their
# pools are tiny (a handful of baskets), so 20,000 draws is already hundreds of
# draws per basket.
N_DRAWS = 20_000

# Scenario 1 override. Its pool is larger (150 baskets), so it needs far more
# draws for the uniformity check to be meaningful rather than dominated by
# small-sample variance. 100,000 over ~146 available baskets is ~685 draws per
# basket, plenty for a stable frequency table. (Raising the global N_DRAWS
# instead would only slow down Scenarios 2 and 3 for no benefit.)
N_DRAWS_SCENARIO_1 = 100_000


def draw_exact(arr, visited_baskets, rng):
    """The reference's exact basket-selection method, isolated.

    Mirrors ``walk_lib.run_walk``: subtract visited, sort into a canonical
    array, uniform ``rng.choice``. Returns the basket, or ``None`` if nothing is
    available.
    """
    available = set(arr.tolist()) - visited_baskets
    if not available:
        return None
    avail_arr = np.array(sorted(available))
    return int(rng.choice(avail_arr))


def draw_fast(arr, visited_baskets, rng):
    """The new full-scale basket-selection method, isolated.

    Byte-for-byte the same decision logic as the basket-selection block in
    ``walk_lib_fullscale.run_walk`` (fast rejection path above
    ``FAST_PATH_MIN_BASKETS``, exact fallback at or below it, and the
    exhausted-rejection fallback). Returns the basket, or ``None`` if nothing is
    available.
    """
    if arr is None or len(arr) == 0:
        return None

    if len(arr) > FAST_PATH_MIN_BASKETS:
        for _ in range(MAX_REJECTION_ATTEMPTS):
            cand = int(arr[rng.integers(0, len(arr))])
            if cand not in visited_baskets:
                return cand
        # Exhausted-rejection fallback (should never fire given the size guarantee).
        available = set(arr.tolist()) - visited_baskets
        if not available:
            return None
        return int(rng.choice(np.array(sorted(available))))
    else:
        available = set(arr.tolist()) - visited_baskets
        if not available:
            return None
        avail_arr = np.array(sorted(available))
        return int(rng.choice(avail_arr))


def empirical_counts(draw_fn, arr, visited_baskets, n_draws, seed):
    """Draw ``n_draws`` baskets with ``draw_fn`` and return a {basket: count} table."""
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(n_draws):
        b = draw_fn(arr, visited_baskets, rng)
        counts[b] = counts.get(b, 0) + 1
    return counts


def check_uniform_over_available(arr, visited_baskets, label, n_draws=N_DRAWS):
    """Compare exact vs fast empirical distributions over the available baskets.

    ``n_draws`` is per-scenario so a larger pool can be given proportionally
    more draws (otherwise a perfectly uniform sampler fails on variance alone).
    Prints a per-scenario PASS/FAIL. PASS requires: both methods only ever draw
    baskets that are actually available (in ``arr`` and not visited), both cover
    the full available set, and both look uniform within ``MAX_REL_DEVIATION``.
    """
    available = sorted(set(arr.tolist()) - visited_baskets)
    print(f"\n[{label}]")
    print(f"  array size = {len(arr)}, visited = {len(visited_baskets)}, "
          f"available = {len(available)} "
          f"(fast path: {'yes' if len(arr) > FAST_PATH_MIN_BASKETS else 'no'}), "
          f"draws = {n_draws:,}")

    exact_counts = empirical_counts(draw_exact, arr, visited_baskets, n_draws, seed=1234)
    fast_counts = empirical_counts(draw_fast, arr, visited_baskets, n_draws, seed=5678)

    expected = n_draws / len(available)
    ok = True

    for name, counts in (("exact", exact_counts), ("fast", fast_counts)):
        drawn = set(counts)
        # Never draw something unavailable.
        illegal = drawn - set(available)
        if illegal:
            print(f"  FAIL: {name} drew unavailable baskets: {sorted(illegal)[:5]}...")
            ok = False
        # Cover the whole available set (with these per-scenario draw counts,
        # every available basket should appear at least once).
        missing = set(available) - drawn
        if missing:
            print(f"  FAIL: {name} never drew {len(missing)} available basket(s)")
            ok = False
        # Uniformity: worst relative deviation from the expected frequency.
        if drawn <= set(available):
            worst = max(abs(counts.get(b, 0) - expected) / expected for b in available)
            flag = "ok" if worst <= MAX_REL_DEVIATION else "TOO SKEWED"
            print(f"  {name:5s}: max rel deviation = {worst:.3f}  ({flag})")
            if worst > MAX_REL_DEVIATION:
                ok = False

    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def check_nothing_available(arr, visited_baskets, label):
    """Both methods must signal 'nothing available' (return None), not loop/error."""
    print(f"\n[{label}]")
    print(f"  array size = {len(arr)}, visited covers all = "
          f"{set(arr.tolist()) <= visited_baskets}")
    rng = np.random.default_rng(999)
    exact_res = draw_exact(arr, visited_baskets, rng)
    fast_res = draw_fast(arr, visited_baskets, rng)
    ok = (exact_res is None) and (fast_res is None)
    print(f"  exact -> {exact_res!r}, fast -> {fast_res!r}")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 64)
    print("Validating walk_lib_fullscale basket selection vs the exact method")
    print(f"FAST_PATH_MIN_BASKETS = {FAST_PATH_MIN_BASKETS}, "
          f"MAX_REJECTION_ATTEMPTS = {MAX_REJECTION_ATTEMPTS}")
    print("=" * 64)

    results = []

    # Scenario 1: popular product (fast path). 150 IDs is 3x FAST_PATH_MIN_BASKETS,
    # so this unambiguously exercises the rejection-sampling branch. 100,000 draws
    # (~685 per basket) keeps the uniformity check meaningful rather than
    # variance-dominated. Visited indices span first/early/middle/last.
    rng_setup = np.random.default_rng(0)
    popular_arr = np.sort(
        rng_setup.choice(np.arange(1_000_000), size=150, replace=False)
    ).astype(np.int32)
    popular_visited = set(int(x) for x in popular_arr[[0, 10, 75, 149]])
    results.append(check_uniform_over_available(
        popular_arr, popular_visited, "Scenario 1: popular product (fast path)",
        n_draws=N_DRAWS_SCENARIO_1))

    # Scenario 2: thin product (exact fallback). 6-8 synthetic IDs, a couple
    # already visited. Correctly calibrated at the default N_DRAWS.
    thin_arr = np.array([11, 22, 33, 44, 55, 66, 77], dtype=np.int32)
    thin_visited = {22, 66}
    results.append(check_uniform_over_available(
        thin_arr, thin_visited, "Scenario 2: thin product (exact fallback)"))

    # Scenario 3: thin product with EVERY basket already visited -> nothing
    # available. Both methods must return None cleanly.
    full_visited_arr = np.array([101, 202, 303, 404, 505], dtype=np.int32)
    full_visited = {101, 202, 303, 404, 505}
    results.append(check_nothing_available(
        full_visited_arr, full_visited,
        "Scenario 3: thin product, all baskets visited (nothing available)"))

    print("\n" + "=" * 64)
    overall = "PASS" if all(results) else "FAIL"
    print(f"OVERALL: {overall}  ({sum(results)}/{len(results)} scenarios passed)")
    print("=" * 64)


if __name__ == "__main__":
    main()
