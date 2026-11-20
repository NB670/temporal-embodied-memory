"""Uncertainty estimates that respect episode clustering.

All questions in an episode share one video, so they are not independent.
We use a cluster (episode) bootstrap: resample episodes with replacement,
keep all questions of each sampled episode, and recompute the pooled
question-level mean. Paired method comparisons bootstrap the per-question
difference the same way.

Implementation: with C episodes, per-episode sums s_c and counts n_c, a
replicate with episode multiplicities w (multinomial, sum C) has mean
(w·s) / (w·n). Drawing W (reps x C) at once makes this a matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n_questions: int
    n_episodes: int
    p_value: float | None = None  # two-sided, for differences vs 0


def _weights(n_clusters: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(reps, n_clusters))
    W = np.zeros((reps, n_clusters))
    np.add.at(W, (np.repeat(np.arange(reps), n_clusters), idx.ravel()), 1)
    return W


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, reps: int = 5000,
                      seed: int = 0, ci: float = 0.95, test_zero: bool = False) -> Estimate:
    """Mean of ``values`` with a percentile CI from an episode bootstrap.

    The same ``seed`` gives the same resampled episodes for any metric over
    the same cluster set, so CIs of different methods use common replicates.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0, 0)
    df = pd.DataFrame({"v": values, "c": clusters}).groupby("c", sort=True).v.agg(["sum", "count"])
    s, n = df["sum"].to_numpy(), df["count"].to_numpy()
    W = _weights(len(df), reps, seed)
    boot = (W @ s) / (W @ n)
    a = (1 - ci) / 2
    p = None
    if test_zero:
        p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return Estimate(float(s.sum() / n.sum()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a)),
                    int(values.size), int(len(df)), p)


def paired_difference(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, **kw) -> Estimate:
    """Bootstrap estimate of mean(a - b) over paired questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    return cluster_bootstrap(a - b, clusters, test_zero=True, **kw)
