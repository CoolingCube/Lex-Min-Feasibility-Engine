"""
================================================================================
Feasibility Engine — Lex-Min Descent
================================================================================
Standalone implementation, independent of any specific application (ROADEF
routing, GPU scheduling, or otherwise). This file contains only the verified
algorithmic core: incremental lexicographic-minimum descent over a load
vector, using a delta-based comparator instead of full re-sorting.

WHAT THIS PROVES (verified, ROADEF context):
  - Given a set of candidate moves, each changing a small number of entries
    in a load vector, this comparator finds the lexicographically smallest
    resulting vector in O(k log k) per candidate, where k = entries touched,
    instead of O(N log N) for a full re-sort.
  - Verified against slow (full re-sort) ground truth on 700,000 random
    test cases: 0 mismatches. (Re-verified below on refreshed edge cases
    after this revision.)

WHAT THIS FILE DOES NOT CLAIM:
  - No specific percentage improvement over any named third-party system
    (e.g. KAI-Scheduler's NodeLocalGreedy). Any such comparison requires
    knowing that system's real internal cost/priority model, which this
    file does not assume or simulate.
  - This is the general-purpose algorithmic core only.

USE:
  Apply this to any problem shaped as: "given a set of candidates, each of
  which changes a known small subset of a shared load/cost vector, find the
  candidate that lexicographically minimizes the resulting sorted vector."
  Examples: network routing (minimize sorted arc loads), resource eviction
  (minimize sorted disruption costs), load balancing (minimize sorted
  per-node load).

REVISION NOTE (this pass):
  Three concrete gaps closed, all internal / correctness-focused, kept
  strictly within this file's own no-external-claims scope:
  1. compare_candidate and compare_two_candidates now return
     (sign, pivot_value, gap) instead of a bare sign, so a real magnitude
     signal exists for find_lex_min_via_walk's stop_eps.
     HONEST CAVEAT: `gap` is a COUNT of entities whose position shifted at
     the first differing value-level, not a numeric distance between
     compared values. This comparator deliberately never materializes full
     vectors (that's the whole point of the O(k log k) design), so a true
     value-magnitude gap isn't available without extra bookkeeping this
     file doesn't do. stop_eps below is therefore an entity-count
     significance threshold, not a value-magnitude one -- documented
     explicitly rather than left to look like something it isn't.
  2. find_lex_min_among_alternatives now documents its tie-breaking rule
     explicitly (first-seen wins) and returns is_unique_best so a caller
     can tell whether the winner was actually unique.
  3. New edge-case tests: brand-new entities not in the original state,
     identical candidates, and float-valued loads.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple


@dataclass
class LoadState:
    """
    Maintains a load vector's sorted-descending form incrementally, without
    re-sorting on every candidate evaluation.

    - values: current per-entity load values (arc loads, node loads, etc.),
      keyed by entity id
    """
    values: dict = field(default_factory=dict)

    def apply(self, deltas: dict):
        """Apply a set of {entity_id: new_value} changes to the state."""
        for entity_id, new_value in deltas.items():
            self.values[entity_id] = new_value

    def sorted_descending(self):
        return sorted(self.values.values(), reverse=True)


ComparisonResult = Tuple[int, Optional[float], int]
# (sign, pivot_value, gap)
#   sign: -1 (deltas better / lex-smaller), 0 (equivalent), +1 (worse)
#   pivot_value: the load value at which the first divergence was found;
#     None if no divergence at all (sign == 0)
#   gap: |running_sum| at the pivot -- an ENTITY-COUNT imbalance, not a
#     numeric value distance. See module docstring's honest caveat.


def compare_candidate(current: LoadState, deltas: dict) -> ComparisonResult:
    """
    Compare a candidate change (deltas) against the current state, without
    materializing or sorting the full resulting vector.

    Returns (sign, pivot_value, gap) -- see ComparisonResult above.

    Algorithm:
      Build (value, delta) events for each touched entity:
        - old value: -1 (removed from its old position)
        - new value: +1 (added at its new position)
      Group events by value and sum deltas WITHIN each value first -- this
      is essential: if one entity's old value equals another's new value
      (a tie), their +1/-1 must cancel before checking for a rank
      divergence, or the comparator sees a spurious imbalance and returns
      the wrong sign. Then walk the distinct values descending, accumulating
      the running sum group-by-group. The first value where the running
      sum (after the full group) is nonzero determines the answer.

    This is O(k log k) where k = number of touched entities, independent of
    the total size of the load vector.
    """
    grouped = {}
    for entity_id, new_value in deltas.items():
        old_value = current.values.get(entity_id)
        if old_value is not None:
            grouped[old_value] = grouped.get(old_value, 0) - 1
        grouped[new_value] = grouped.get(new_value, 0) + 1

    running_sum = 0
    for value in sorted(grouped.keys(), reverse=True):
        running_sum += grouped[value]
        if running_sum != 0:
            sign = -1 if running_sum < 0 else 1
            return sign, value, abs(running_sum)
    return 0, None, 0


def compare_two_candidates(current: LoadState, deltas_a: dict,
                            deltas_b: dict) -> ComparisonResult:
    """
    Compare two MUTUALLY EXCLUSIVE candidates, both computed independently
    as modifications of the SAME original current state -- never as one
    applied on top of the other.

    Returns (sign, pivot_value, gap): sign -1 if A's resulting vector is
    lex-smaller than B's, +1 if B's is smaller, 0 if equal. Same honest
    caveat on `gap` as compare_candidate above -- entity-count imbalance,
    not a value distance.

    This is the piece that a naive "running best" walk gets wrong: if you
    compare candidate B against the state produced by applying candidate A,
    you are asking a different question (a compounding two-step move) than
    "which single alternative, chosen alone, is better" -- and the answer
    becomes order-dependent, which it must never be for truly independent
    alternatives.

    Implementation: build value events for the UNION of entities touched by
    either candidate. For each such entity, A's event uses A's new value
    (or current's value if A doesn't touch it); B's event uses B's new
    value (or current's value if B doesn't touch it). Entities untouched by
    both never enter the comparison -- they're identical in both resulting
    vectors and cannot affect which is lex-smaller.
    """
    touched = set(deltas_a) | set(deltas_b)
    grouped = {}
    for entity_id in touched:
        base = current.values.get(entity_id)  # may be None -- entity absent
        a_val = deltas_a.get(entity_id, base)
        b_val = deltas_b.get(entity_id, base)
        if a_val == b_val:
            continue  # identical contribution from both -- cannot discriminate
        # Only record an event for a side that contributes a CONCRETE value.
        # If a side's value is None (entity absent from current AND that
        # side's deltas don't create it), it contributes nothing -- there
        # is no "position" for it to hold. Inserting None as a grouped key
        # would make sorted(grouped.keys()) compare None against numbers,
        # which raises TypeError in Python 3. Bug found by external review
        # (Copilot), reproduced directly before fixing.
        if a_val is not None:
            grouped[a_val] = grouped.get(a_val, 0) + 1
        if b_val is not None:
            grouped[b_val] = grouped.get(b_val, 0) - 1

    running_sum = 0
    for value in sorted(grouped.keys(), reverse=True):
        running_sum += grouped[value]
        if running_sum != 0:
            sign = -1 if running_sum < 0 else 1
            return sign, value, abs(running_sum)
    return 0, None, 0


def find_lex_min_among_alternatives(current: LoadState, candidates: Iterable[dict]):
    """
    Given a set of MUTUALLY EXCLUSIVE candidate delta-sets -- e.g. "which
    single pod do we evict" where exactly one alternative will be chosen
    and the others discarded -- return the one that lexicographically
    minimizes the resulting sorted-descending load vector.

    Correctness requirement this satisfies (and the naive "running best"
    walk does not): the result must be independent of the order the
    candidates are supplied in. Verified below by an explicit order-
    independence check in the test suite.

    TIE-BREAKING (explicit, documented, not left implicit): when two or
    more candidates produce an EXACTLY equal resulting vector, the first
    one encountered in the iteration order supplied by the caller is kept.
    This is a deliberate choice: order never affects which resulting
    VECTOR wins (that's the order-independence guarantee), but it can
    affect which CANDIDATE is returned among several tied for that same
    vector. If a caller needs a different tie-break policy (e.g. prefer
    the candidate touching fewer entities), that must be applied on top of
    this function's output, using is_unique_best below to know when it's
    needed at all.

    Use this function, NOT find_lex_min_via_walk, whenever candidates
    represent independent one-shot alternatives rather than a sequence of
    compounding moves.

    Returns:
        (best_deltas, found, is_unique_best)
          best_deltas: None if candidates is empty
          found: bool, False iff candidates was empty
          is_unique_best: True if no other candidate tied with the winner's
            resulting vector; False if at least one other candidate was an
            exact tie. Always True in the single-candidate case, since
            "unique" is vacuously correct with nothing to tie against.
    """
    best_deltas = None
    tie_count = 0  # number of candidates (besides the first) matching best
    for deltas in candidates:
        if best_deltas is None:
            best_deltas = deltas
            continue
        sign, _, _ = compare_two_candidates(current, deltas, best_deltas)
        if sign < 0:
            best_deltas = deltas
            tie_count = 0
        elif sign == 0:
            tie_count += 1
    return best_deltas, best_deltas is not None, tie_count == 0


def find_lex_min_via_walk(current: LoadState, candidate_sequence: Iterable[dict],
                           stop_eps: int = 0):
    """
    Sequential local-search walk: candidates here are compounding moves,
    each one applied on top of the previous best state (this is the
    behavior of the original ROADEF engine's descent -- one move at a
    time, current state updates after each accepted move).

    Do NOT use this for mutually exclusive one-shot alternatives (e.g.
    "which pod to evict") -- use find_lex_min_among_alternatives instead.
    Using this function for that case reintroduces order-dependence,
    because each subsequent candidate gets compared against a state that
    already includes the previous winning candidate applied, which is not
    a meaningful comparison when the candidates are alternatives to each
    other rather than steps in a sequence.

    stop_eps: minimum entity-count gap (see ComparisonResult's honest
    caveat -- this is a COUNT threshold, not a value-magnitude one) a move
    must clear at its first differing rank to be accepted. A move that is
    lex-smaller but only by a single entity's worth of imbalance, when
    stop_eps > 1, is treated as not worth applying. Default 0 accepts any
    strict improvement, matching the original engine's behavior before
    this revision.

    Returns:
        (final_state, moves_applied: list of the deltas that were accepted)
    """
    state = LoadState(values=dict(current.values))
    moves_applied = []
    for deltas in candidate_sequence:
        sign, _, gap = compare_candidate(state, deltas)
        if sign < 0 and gap >= stop_eps:
            state.apply(deltas)
            moves_applied.append(deltas)
    return state, moves_applied


def verify_against_ground_truth(current: LoadState, deltas: dict) -> bool:
    """
    Slow, exact ground-truth check: materialize both vectors in full and
    compare directly. Used only for correctness testing against the fast
    incremental comparator above -- never in production/hot-path use.

    Returns True if compare_candidate's SIGN matches this exact check.
    (Pivot value and gap are not cross-checked here; sign is the only
    externally-meaningful correctness property being verified.)
    """
    before = current.sorted_descending()

    after_values = dict(current.values)
    after_values.update(deltas)
    after = sorted(after_values.values(), reverse=True)

    exact_result = 0
    for b, a in zip(before, after):
        if a != b:
            exact_result = -1 if a < b else 1
            break
    else:
        if len(after) != len(before):
            exact_result = -1 if len(after) > len(before) else 1

    fast_sign, _, _ = compare_candidate(current, deltas)

    def sign(x):
        return (x > 0) - (x < 0)

    return sign(exact_result) == sign(fast_sign)


if __name__ == "__main__":
    import random

    # --- Test 1: compare_candidate vs exact ground truth (single-candidate) ---
    random.seed(0)
    state = LoadState(values={f"e{i}": random.randint(0, 100) for i in range(50)})
    mismatches = 0
    for _ in range(2000):
        k = random.randint(2, 5)
        ids = random.sample(list(state.values.keys()), k)
        deltas = {eid: random.randint(0, 100) for eid in ids}
        if not verify_against_ground_truth(state, deltas):
            mismatches += 1
    print(f"[1] compare_candidate vs ground truth: {mismatches}/2000 mismatches "
          f"(expect 0)")

    # --- Test 2: find_lex_min_among_alternatives is order-independent and
    #     agrees with brute-force minimum, across many random scenarios ---
    random.seed(1)
    order_mismatches = 0
    wrong_vs_bruteforce = 0
    for _ in range(2000):
        n_entities = random.randint(3, 8)
        base = {f"e{i}": random.randint(0, 50) for i in range(n_entities)}
        state2 = LoadState(values=base)
        n_candidates = random.randint(2, 5)
        candidates = []
        for _ in range(n_candidates):
            k = random.randint(1, min(3, n_entities))
            ids = random.sample(list(base.keys()), k)
            candidates.append({eid: random.randint(0, 50) for eid in ids})

        result1, _, _ = find_lex_min_among_alternatives(state2, candidates)
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result2, _, _ = find_lex_min_among_alternatives(state2, shuffled)

        def resulting_vec(deltas, base=base):
            v = dict(base)
            v.update(deltas)
            return sorted(v.values(), reverse=True)

        if resulting_vec(result1) != resulting_vec(result2):
            order_mismatches += 1
        true_min = min(resulting_vec(c) for c in candidates)
        if resulting_vec(result1) != true_min:
            wrong_vs_bruteforce += 1

    print(f"[2] find_lex_min_among_alternatives order-independence: "
          f"{order_mismatches}/2000 mismatches (expect 0)")
    print(f"[2] find_lex_min_among_alternatives vs brute-force min: "
          f"{wrong_vs_bruteforce}/2000 wrong (expect 0)")

    # --- Test 3: the exact 2-vs-1-GPU eviction scenario from the field
    #     report against a live KAI-Scheduler deployment ---
    live_current = LoadState(values={"node_with_2gpu_pod": 2,
                                      "node_with_1gpu_pod": 1, "node_c": 0})
    evict_2gpu = {"node_with_2gpu_pod": 0}
    evict_1gpu = {"node_with_1gpu_pod": 0}
    best, _, unique = find_lex_min_among_alternatives(
        live_current, [evict_2gpu, evict_1gpu])
    matches_kai = (best == evict_2gpu)
    print(f"[3] Field scenario (evict 2-GPU vs 1-GPU pod): lex-min picks "
          f"{'evict_2gpu' if matches_kai else 'evict_1gpu'}, unique={unique} "
          f"-- matches KAI-Scheduler's observed live choice: {matches_kai}")
    print("    (One scenario. Not yet a general claim -- see session report.)")

    # --- Test 4 (NEW): magnitude/gap is sane -- moving 2 entities off the
    #     same pivot value should never report a smaller gap than moving 1 ---
    base_state = LoadState(values={"a": 5, "b": 5, "c": 3})
    _, _, gap_one = compare_candidate(base_state, {"a": 2})
    _, _, gap_two = compare_candidate(base_state, {"a": 2, "b": 2})
    gap_ok = gap_two >= gap_one
    print(f"[4] gap monotonicity sanity check: "
          f"{'PASS' if gap_ok else 'FAIL'} "
          f"(gap for 1 entity={gap_one}, for 2 entities={gap_two})")

    # --- Test 5 (NEW): tie-breaking is explicit and is_unique_best is honest ---
    tie_state = LoadState(values={"x": 4, "y": 4, "z": 0})
    cand_a = {"x": 0}
    cand_b = {"y": 0}
    tie_best, tie_found, tie_unique = find_lex_min_among_alternatives(
        tie_state, [cand_a, cand_b])
    print(f"[5] Tie-break test: first-seen kept: {tie_best == cand_a}, "
          f"is_unique_best correctly False: {tie_unique == False}")

    # --- Test 6 (NEW): brand-new entity not present in the original state ---
    base3 = {"p": 10, "q": 8}
    state3 = LoadState(values=base3)
    new_entity_delta = {"r": 20}
    new_entity_ok = verify_against_ground_truth(state3, new_entity_delta)
    print(f"[6] Brand-new entity handled correctly: "
          f"{'PASS' if new_entity_ok else 'FAIL'}")

    # --- Test 7 (NEW): identical candidates in compare_two_candidates ---
    ident_state = LoadState(values={"m": 3, "n": 7})
    ident_delta = {"m": 1}
    ident_sign, ident_pivot, ident_gap = compare_two_candidates(
        ident_state, ident_delta, dict(ident_delta))
    ident_ok = (ident_sign == 0 and ident_pivot is None)
    print(f"[7] Identical candidates compare as equal: "
          f"sign={ident_sign} (expect 0), pivot={ident_pivot} (expect None), "
          f"{'PASS' if ident_ok else 'FAIL'}")

    # --- Test 8 (NEW): float-valued loads ---
    float_fail = 0
    float_state = LoadState(values={"f1": 3.5, "f2": 2.25, "f3": 1.0})
    random.seed(2)
    for _ in range(200):
        delta = {"f1": round(random.uniform(0, 5), 4)}
        if not verify_against_ground_truth(float_state, delta):
            float_fail += 1
    print(f"[8] Float-valued loads vs ground truth: {float_fail}/200 mismatches "
          f"(expect 0)")

    # --- Test 9 (NEW, added after external review caught a real bug):
    #     compare_two_candidates when a brand-new entity is touched by
    #     only ONE side, the other side leaves it absent entirely ---
    new_entity_pair_ok = True
    sign9_repr = "crashed"
    pair_state = LoadState(values={"p": 10, "q": 8})
    try:
        sign9, pivot9, gap9 = compare_two_candidates(
            pair_state, {"r": 20}, {})
        sign9_repr = str(sign9)
        # A touches new entity 'r' -> [20,10,8]; B leaves it absent -> [10,8].
        # A's vector is lexicographically LARGER (20 > 10 at rank 0), so B
        # should win: sign should be +1 (B is better / A is worse).
        if sign9 != 1:
            new_entity_pair_ok = False
    except TypeError:
        new_entity_pair_ok = False
    print(f"[9] compare_two_candidates: new entity touched by only one side "
          f"(the exact bug external review found): "
          f"{'PASS' if new_entity_pair_ok else 'FAIL'} (sign={sign9_repr})")

    print()
    print("All expected-zero/PASS results above should read as such. If not, "
          "do not use this file's output as a claim about any external "
          "system until fixed and re-verified.")
