"""POST-HOC method (added after the main results): semantic top-k with temporal
non-maximum suppression (NMS).

It is not part of the pre-specified comparison. We added it after observing
that TNR's temporal *context* hurts because neighbouring frames are redundant
under contiguous labels. It tests the inverse hypothesis: does temporal
*diversity* help?

Greedy over frames in descending similarity: accept a frame unless it lies
within ``gap`` frames of an already accepted one. If fewer than k frames
survive suppression, backfill with the best remaining frames in semantic
order. ``gap = 0`` is exactly semantic top-k.
"""

from __future__ import annotations

import numpy as np

from src.memory.base import Retriever, stable_topk
from src.memory.frame_store import EpisodeMemory


class SemanticNMSRetriever(Retriever):
    name = "semantic_nms"

    def __init__(self, gap: int = 2) -> None:
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.gap = gap

    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int):
        if query_emb is None:
            raise ValueError("needs a query embedding")
        s = memory.similarity(query_emb)
        order = stable_topk(s, int(np.isfinite(s).sum())).tolist()
        chosen: list[int] = []
        for t in order:
            if len(chosen) == k:
                break
            if all(abs(t - c) > self.gap for c in chosen):
                chosen.append(t)
        for t in order:  # backfill
            if len(chosen) == k:
                break
            if t not in chosen:
                chosen.append(t)
        return chosen, [float(s[t]) for t in chosen]

    def config(self) -> dict:
        return {"gap": self.gap}
