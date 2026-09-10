# Lex-Min Feasibility Engine

A small, general-purpose comparator for a specific, common shape of problem:

> Given a set of candidate moves, each of which changes a known, small subset of a shared load/cost vector — which candidate results in the **lexicographically smallest sorted vector**?

Applies to problems such as: network routing (minimize sorted arc loads), resource eviction (minimize sorted disruption costs), load balancing (minimize sorted per-node load).

## What it does

Most implementations of this comparison re-sort the entire vector for every candidate — `O(N log N)` per candidate, where `N` is the total number of entities. This engine instead looks only at the entities a candidate actually touches — `O(k log k)`, where `k` is the number of touched entities — and produces the same answer.

It also addresses a subtler issue: comparing two independent candidates fairly. A naive approach compares candidate B against the state *after* applying candidate A, which implicitly asks a different question (a compounding two-step move) and makes the result depend on comparison order. This engine compares both candidates against the same original state, so results are order-independent.

## Verified

The file's own test suite (`python lexmin_engine_v2.py`) checks:
- The fast comparator against a slow, exact ground-truth re-sort, across thousands of random cases.
- Order-independence: shuffling candidate order never changes the result.
- Explicit tie-breaking behavior, with an `is_unique_best` flag distinguishing a genuine win from an arbitrarily-broken tie.
- Edge cases: brand-new entities, identical candidates, float-valued loads.

Scope: this is a comparator, not a full solver. It does not claim a specific performance advantage over any named third-party scheduler, and does not solve feasibility in the general case.

## Relation to KAI-Scheduler

- The engine's test suite includes one field-observed scenario: comparing the eviction cost of a 2-GPU pod versus a 1-GPU pod against a live KAI-Scheduler deployment's observed choice. One data point, not a general claim.
- [PR #1995](https://github.com/kai-scheduler/KAI-Scheduler/pull/1995) (merged) adds a configurable, resource-aware tiebreak for `JobOrderFn` and a new `VictimOrderFn` extension point, written independently in KAI's Go codebase. It follows the same underlying principle as this engine — comparing candidates by their effect on shared resource state rather than by an identifier with no scheduling meaning — without importing this code directly.
- [Issue #2120](https://github.com/kai-scheduler/KAI-Scheduler/issues/2120) (open) concerns `SubGroupOrderFn`'s fallback to alphabetical subgroup name when its primary comparator ties. This engine's order-independent comparison is one candidate ingredient for a fix; full cluster-wide placement feasibility is a separate, harder problem this engine does not address.

## Where this may extend to GPU scheduling

- Tiebreaks between resource-equivalent candidates: plausible fit.
- Eviction/preemption victim selection: close to the engine's own tested scenario.
- Full bin-packing / placement feasibility: not a fit — a different problem class (whether a given shape of demand fits fragmented capacity), not addressed here.

## Usage

```python
from lexmin_engine_v2 import LoadState, find_lex_min_among_alternatives

state = LoadState(values={"node_a": 2, "node_b": 1, "node_c": 0})
candidates = [{"node_a": 0}, {"node_b": 0}]  # candidate eviction targets

best, found, is_unique = find_lex_min_among_alternatives(state, candidates)
```

No dependencies beyond the Python standard library.
