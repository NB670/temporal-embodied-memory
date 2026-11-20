import numpy as np
import pytest

from src.memory.random_retriever import RandomRetriever
from src.memory.semantic_retriever import SemanticRetriever
from src.memory.temporal_retriever import TemporalNeighborhoodRetriever as TNR
from src.memory.uniform_retriever import UniformRetriever
from tests.conftest import make_memory

RNG = np.random.default_rng(123)


def rand_memory(T=96, invalid=()):
    valid = np.ones(T, bool)
    valid[list(invalid)] = False
    return make_memory(RNG.uniform(-0.2, 0.4, T), valid=valid)


ALL = [UniformRetriever(), RandomRetriever(0), SemanticRetriever(), TNR(0), TNR(1), TNR(2), TNR(4),
       TNR(2, anchor_frac=0.25), TNR(4, anchor_frac=1.0)]


@pytest.mark.parametrize("ret", ALL, ids=lambda r: f"{r.name}{r.config()}")
@pytest.mark.parametrize("k", [1, 4, 8, 16, 32, 96])
def test_budget_unique_valid_chronological(ret, k):
    mem, q = rand_memory(invalid=[0, 5, 50])
    r = ret.retrieve(mem, q, k)
    k_eff = min(k, 93)
    assert len(r.frame_indices) == k_eff == len(set(r.frame_indices))
    assert r.frame_indices == sorted(r.frame_indices)
    assert not {0, 5, 50} & set(r.frame_indices)
    assert all(0 <= i < 96 for i in r.frame_indices)
    assert sorted(r.ranking) == r.frame_indices


@pytest.mark.parametrize("ret", ALL, ids=lambda r: f"{r.name}{r.config()}")
def test_small_episode(ret):
    mem, q = make_memory([0.1, 0.5, 0.3])
    assert ret.retrieve(mem, q, 8).frame_indices == [0, 1, 2]


def test_uniform_positions():
    mem, q = rand_memory()
    assert UniformRetriever().retrieve(mem, q, 8).frame_indices == [6, 18, 30, 42, 54, 66, 78, 90]


def test_random_reproducible_and_seed_dependent():
    mem, q = rand_memory()
    a = RandomRetriever(0).retrieve(mem, q, 8).frame_indices
    assert a == RandomRetriever(0).retrieve(mem, q, 8).frame_indices
    assert a != RandomRetriever(1).retrieve(mem, q, 8).frame_indices


@pytest.mark.parametrize("k", [4, 8, 16, 32])
def test_radius_zero_equals_semantic(k):
    for _ in range(20):
        mem, q = rand_memory()
        assert TNR(0).retrieve(mem, q, k).frame_indices == SemanticRetriever().retrieve(mem, q, k).frame_indices


def test_tnr_neighbourhood_and_priority():
    s = np.full(20, 0.0)
    s[10], s[3] = 0.9, 0.8          # anchors for k=4 (2 anchors)
    s[11], s[9] = 0.2, 0.1          # neighbours of 10: 11 preferred over 9 by score
    s[2], s[4] = 0.05, 0.3
    mem, q = make_memory(s)
    r = TNR(radius=1).retrieve(mem, q, 4)
    assert r.ranking[:2] == [10, 3]
    assert set(r.ranking[2:]) == {4, 11}          # distance-1 neighbours, best scores first
    assert r.frame_indices == [3, 4, 10, 11]


def test_tnr_overlap_dedup_and_backfill():
    s = np.linspace(0, 1, 10)[::-1].copy()        # frames 0,1 are best and adjacent
    mem, q = make_memory(s)
    r = TNR(radius=1).retrieve(mem, q, 6)         # anchors 0,1 -> neighbour only {2}
    assert r.ranking[:3] == [0, 1, 2]
    assert r.ranking[3:] == [3, 4, 5]              # semantic backfill
    assert len(set(r.frame_indices)) == 6


def test_tnr_boundaries():
    s = np.zeros(12)
    s[0], s[11] = 0.9, 0.8
    mem, q = make_memory(s)
    r = TNR(radius=2, anchor_frac=1 / 3).retrieve(mem, q, 6)  # 2 anchors + 4 neighbours
    assert set(r.frame_indices) == {0, 1, 2, 9, 10, 11}


def test_tnr_skips_invalid_neighbours():
    s = np.zeros(10)
    s[5] = 0.9
    mem, q = make_memory(s, valid=[True] * 4 + [False, True, False] + [True] * 3)
    r = TNR(radius=2, anchor_frac=0.25).retrieve(mem, q, 4)
    assert 4 not in r.frame_indices and 6 not in r.frame_indices
    assert {5, 3, 7} <= set(r.frame_indices)


def test_tnr_deterministic():
    mem, q = rand_memory()
    assert TNR(2).retrieve(mem, q, 8).ranking == TNR(2).retrieve(mem, q, 8).ranking


def test_tnr_bad_params():
    with pytest.raises(ValueError):
        TNR(-1)
    with pytest.raises(ValueError):
        TNR(1, anchor_frac=0)


# ---- post-hoc diversity retriever ----
from src.memory.diversity_retriever import SemanticNMSRetriever  # noqa: E402


@pytest.mark.parametrize("gap", [0, 1, 2, 4, 50])
@pytest.mark.parametrize("k", [1, 8, 32, 96])
def test_nms_budget_valid(gap, k):
    mem, q = rand_memory(invalid=[3])
    r = SemanticNMSRetriever(gap).retrieve(mem, q, k)
    assert len(set(r.frame_indices)) == min(k, 95) and 3 not in r.frame_indices


def test_nms_gap_zero_is_semantic():
    mem, q = rand_memory()
    assert SemanticNMSRetriever(0).retrieve(mem, q, 8).frame_indices == SemanticRetriever().retrieve(mem, q, 8).frame_indices


def test_nms_suppresses_neighbours():
    s = np.zeros(20); s[10], s[11], s[3] = 0.9, 0.85, 0.5
    mem, q = make_memory(s)
    assert SemanticNMSRetriever(1).retrieve(mem, q, 2).frame_indices == [3, 10]
