"""Baseline A: uniform sampling.

Select k frames evenly spaced over the episode, ignoring the query. This is
the frame-selection strategy the FindingDory paper uses for every VLM
baseline (re-applied here inside the 96-frame videos).

With V valid frames we split them into k equal-width bins and take the centre
of each bin: position_j = floor((j + 0.5) * V / k), j = 0..k-1. Positions are
distinct whenever k <= V. For T = V = 96, k = 8 this gives 6, 18, 30, ..., 90.
"""

from __future__ import annotations

import numpy as np

from src.memory.base import Retriever
from src.memory.frame_store import EpisodeMemory


class UniformRetriever(Retriever):
    name = "uniform"

    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int):
        valid = np.flatnonzero(memory.valid)
        pos = np.floor((np.arange(k) + 0.5) * len(valid) / k).astype(int)
        return valid[pos].tolist(), None
