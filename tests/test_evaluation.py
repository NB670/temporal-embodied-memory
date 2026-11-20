import numpy as np
import pytest

from src.evaluation.metrics import question_metrics
from src.evaluation.statistics import cluster_bootstrap, paired_difference


def test_single_goal_metrics():
    m = question_metrics([1, 5, 9, 20], [[5, 6, 7]])
    assert m == {"any_hit": 1.0, "goal_coverage": 1.0, "all_goals": 1.0, "precision": 0.25, "recall": 1 / 3}


def test_multi_goal_keeps_goal_identity():
    goals = [[1, 2], [10, 11], [20]]
    m = question_metrics([2, 11, 50], goals)
    assert m["goal_coverage"] == pytest.approx(2 / 3) and m["all_goals"] == 0.0 and m["any_hit"] == 1.0
    # flattening would wrongly give full coverage for two frames in one goal
    assert question_metrics([1, 2], goals)["goal_coverage"] == pytest.approx(1 / 3)


def test_overlapping_goals_one_frame_covers_both():
    m = question_metrics([5], [[4, 5], [5, 6]])
    assert m["goal_coverage"] == 1.0 and m["precision"] == 1.0


def test_miss():
    assert question_metrics([0, 1], [[50]])["any_hit"] == 0.0


@pytest.mark.parametrize("goals", [[[-1]], [[]], [[3], [-1]], []])
def test_special_answers_rejected(goals):
    with pytest.raises(ValueError):
        question_metrics([1], goals)


def test_duplicates_rejected():
    with pytest.raises(ValueError):
        question_metrics([1, 1], [[1]])


def test_bootstrap_constant_and_ci_contains_mean():
    c = np.repeat(np.arange(20), 5)
    e = cluster_bootstrap(np.ones(100), c, reps=500)
    assert e.mean == e.lo == e.hi == 1.0 and e.n_episodes == 20
    v = np.random.default_rng(0).random(100)
    e = cluster_bootstrap(v, c, reps=2000)
    assert e.lo <= e.mean <= e.hi


def test_paired_difference_detects_shift_and_null():
    rng = np.random.default_rng(1)
    c = np.repeat(np.arange(30), 10)
    b = rng.random(300)
    d = paired_difference(b + 0.1, b, c, reps=2000)
    assert d.mean == pytest.approx(0.1) and d.p_value < 0.01
    d0 = paired_difference(b, b, c, reps=500)
    assert d0.mean == 0 and d0.p_value == 1.0


def test_bootstrap_is_cluster_level():
    # one huge episode with all the signal: CI must be wide, not question-level narrow
    v = np.r_[np.ones(100), np.zeros(10 * 10)]
    c = np.r_[np.zeros(100, int), np.repeat(np.arange(1, 11), 10)]
    e = cluster_bootstrap(v, c, reps=3000)
    assert e.hi - e.lo > 0.4
