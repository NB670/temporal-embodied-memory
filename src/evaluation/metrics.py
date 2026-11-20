"""Retrieval metrics against FindingDory ground-truth frame lists.

Ground truth for one question is ``goals: list[list[int]]``, one list of
acceptable frame indices per goal, with goal identity preserved. These
metrics apply only to questions where every goal has at least one valid frame
(``answer_kind == "frames"``). ``[[-1]]`` / ``[[]]`` questions have no frame
target and are analysed separately.

Given the selected set S (|S| = k) and union U of all goal lists:

* ``any_hit``       1[S ∩ U ≠ ∅]. For single-goal questions this matches the
                    paper's SFT protocol (a predicted frame is correct if it is
                    in the valid list).
* ``goal_coverage`` (1/G) Σ_g 1[S ∩ G_g ≠ ∅]. Fraction of goals with at least
                    one selected frame. Equals ``any_hit`` when G = 1. One frame
                    may cover two goals whose lists overlap; it counts for both,
                    because both goals' viewpoints are then in memory.
* ``all_goals``     1[every goal covered]. Strict multi-goal success.
* ``precision``     |S ∩ U| / |S|. Biased upward for any method that selects
                    adjacent frames, since labels are contiguous runs.
* ``recall``        |S ∩ U| / |U|. Bounded above by k / |U|.
"""

from __future__ import annotations


def question_metrics(selected: list[int], goals: list[list[int]]) -> dict[str, float]:
    if not goals or any(len(g) == 0 or g == [-1] for g in goals):
        raise ValueError(f"metrics undefined for special answer {goals}")
    if len(selected) == 0:
        raise ValueError("empty selection")
    S = set(selected)
    if len(S) != len(selected):
        raise ValueError("selection contains duplicates")
    U = {i for g in goals for i in g}
    covered = [bool(S & set(g)) for g in goals]
    inter = len(S & U)
    return {
        "any_hit": float(inter > 0),
        "goal_coverage": sum(covered) / len(goals),
        "all_goals": float(all(covered)),
        "precision": inter / len(S),
        "recall": inter / len(U),
    }
