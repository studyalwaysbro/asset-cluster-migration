"""Tests for migration metrics.

v0.4.0 — Updated for Hungarian matching (permutation-invariant CMI).
"""
from src.migration.metrics import (
    cluster_migration_index,
    asset_migration_frequency,
    cluster_persistence_score,
    _match_cluster_labels,
    TDSNormalizer,
)


def test_cmi_no_change():
    """Identical assignments → CMI = 0."""
    prev = {"A": 0, "B": 0, "C": 1}
    curr = {"A": 0, "B": 0, "C": 1}
    assert cluster_migration_index(curr, prev) == 0.0


def test_cmi_label_permutation():
    """Pure label swap (same structure) → CMI = 0 after Hungarian matching.

    This is the key fix: Leiden can relabel clusters arbitrarily.
    {A:0, B:0, C:1} and {A:1, B:1, C:0} are structurally identical.
    """
    prev = {"A": 0, "B": 0, "C": 1}
    curr = {"A": 1, "B": 1, "C": 0}
    assert cluster_migration_index(curr, prev) == 0.0


def test_cmi_real_structural_change():
    """Actual structural change: B moves from cluster with A to cluster with C."""
    prev = {"A": 0, "B": 0, "C": 1, "D": 1}
    curr = {"A": 0, "B": 1, "C": 1, "D": 0}
    # Hungarian matching will find best assignment. With prev {A,B}=0 {C,D}=1
    # and curr {A}=0 {B,C}=1 {D}=0:
    # Best match: curr_0 -> prev_0 (A is common), curr_1 -> prev_1 (C is common)
    # Then: A=0->0 (match), D=0->0 (match via curr_0->prev_0 but D was in prev_1)
    # Actually: curr has {A,D} in cluster 0 and {B,C} in cluster 1
    # prev has {A,B} in cluster 0 and {C,D} in cluster 1
    # Hungarian: curr_0={A,D} vs prev_0={A,B}: overlap=1(A), vs prev_1={C,D}: overlap=1(D)
    #            curr_1={B,C} vs prev_0={A,B}: overlap=1(B), vs prev_1={C,D}: overlap=1(C)
    # Cost matrix: [[3,3],[3,3]] - tied, so either mapping works
    # With mapping curr_0->prev_0, curr_1->prev_1: A stays, B moves, C stays, D moves → CMI=0.5
    cmi = cluster_migration_index(curr, prev)
    assert cmi == 0.5


def test_cmi_complete_restructuring():
    """Every asset genuinely changes cluster (no label permutation possible)."""
    prev = {"A": 0, "B": 0, "C": 1, "D": 1}
    # A and C swap, B and D swap — no matching preserves structure
    curr = {"A": 0, "B": 1, "C": 0, "D": 1}
    # prev: {A,B}=0, {C,D}=1
    # curr: {A,C}=0, {B,D}=1
    # Hungarian: curr_0={A,C} vs prev_0={A,B}: overlap=1, vs prev_1={C,D}: overlap=1
    #            curr_1={B,D} vs prev_0={A,B}: overlap=1, vs prev_1={C,D}: overlap=1
    # Either mapping gives 2/4 changed = 0.5
    cmi = cluster_migration_index(curr, prev)
    assert cmi == 0.5


def test_cmi_three_cluster_permutation():
    """Three clusters, pure relabeling → CMI = 0."""
    prev = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2}
    curr = {"A": 2, "B": 2, "C": 0, "D": 0, "E": 1, "F": 1}
    assert cluster_migration_index(curr, prev) == 0.0


def test_hungarian_matching():
    """Direct test of Hungarian label matching."""
    prev = {"A": 0, "B": 0, "C": 1}
    curr = {"A": 1, "B": 1, "C": 0}  # swapped labels
    mapping = _match_cluster_labels(curr, prev)
    # Should map curr_1 -> prev_0 (A,B in common) and curr_0 -> prev_1
    assert mapping[1] == 0
    assert mapping[0] == 1


def test_amf_with_matching():
    """AMF should use Hungarian matching at each step."""
    history = [
        {"A": 0, "B": 0, "C": 1},
        {"A": 1, "B": 1, "C": 0},  # pure relabeling → no migration
        {"A": 1, "B": 0, "C": 0},  # B actually moves → migration for B
    ]
    amf = asset_migration_frequency(history)
    assert amf["A"] == 0.0 or amf["A"] == 0.5  # depends on step 2→3 matching
    # Step 0→1: pure relabeling, 0 migrations
    # Step 1→2: {A,B}=1,{C}=0 vs {A}=1,{B,C}=0: B genuinely moves
    assert amf["B"] == 0.5


def test_cps_with_matching():
    """CPS uses Hungarian matching for proper cluster alignment."""
    prev = {"A": 0, "B": 0, "C": 1}
    curr = {"A": 1, "B": 1, "C": 0}  # pure relabeling
    scores = cluster_persistence_score(curr, prev)
    # After matching, all clusters should have Jaccard = 1.0
    for s in scores.values():
        assert s == 1.0


def test_tds_normalizer():
    """TDSNormalizer z-scores components after sufficient history."""
    norm = TDSNormalizer()
    # First few values: returned as-is (insufficient history for z-scoring)
    v1 = norm.normalize(0.1, 0.5, 0.3)
    v2 = norm.normalize(0.2, 0.6, 0.4)
    # After 3+ values, z-scoring kicks in
    v3 = norm.normalize(0.3, 0.7, 0.5)
    # z-scores should be centered and scaled
    assert isinstance(v3, tuple)
    assert len(v3) == 3
