"""Tests for answer parsing / classification and subset selection (offline)."""

import pandas as pd
import pytest

from src.data.inspect_dataset import chance_any_hit, contiguous_runs
from src.data.load_dataset import (
    ANSWER_EMPTY, ANSWER_FRAMES, ANSWER_PARTIAL, ANSWER_UNSEEN,
    classify_answer, parse_answer, select_episodes,
)


def test_parse_preserves_goal_identity():
    assert parse_answer("[[1, 2], [5, 6]]") == [[1, 2], [5, 6]]


@pytest.mark.parametrize("bad", ["[1, 2]", "[[96]]", "[[-2]]", "[[1.5]]", "{'a': 1}"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_answer(bad)


@pytest.mark.parametrize("goals,kind", [
    ([[3, 4]], ANSWER_FRAMES),
    ([[1], [2, 3]], ANSWER_FRAMES),
    ([[-1]], ANSWER_UNSEEN),
    ([[]], ANSWER_EMPTY),
    ([[-1], [11, 12]], ANSWER_PARTIAL),
    ([[4], []], ANSWER_PARTIAL),
])
def test_classify(goals, kind):
    assert classify_answer(goals) == kind


def test_contiguous_runs():
    assert contiguous_runs([]) == 0
    assert contiguous_runs([5, 6, 7]) == 1
    assert contiguous_runs([7, 5, 6, 20, 21, 40]) == 3


def test_chance_any_hit():
    assert chance_any_hit(0, 8) == 0.0
    assert chance_any_hit(96, 8) == pytest.approx(1.0)
    assert chance_any_hit(1, 8) == pytest.approx(8 / 96)


def _toy_df():
    rows = [dict(ep_id=f"ep_{i}", num_interactions=2 + i % 5) for i in range(1, 51)]
    return pd.DataFrame(rows + rows)  # duplicate rows = multiple questions per episode


def test_select_episodes_deterministic_and_stratified():
    df = _toy_df()
    a, b = select_episodes(df, 10, seed=0), select_episodes(df, 10, seed=0)
    assert a == b and len(a) == len(set(a)) == 10
    strata = df.drop_duplicates("ep_id").set_index("ep_id").loc[a, "num_interactions"]
    assert strata.value_counts().tolist() == [2] * 5


def test_select_episodes_tops_up_when_not_divisible():
    assert len(select_episodes(_toy_df(), 12, seed=1)) == 12
