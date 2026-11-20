"""Temporal-Neighborhood Retrieval (TNR), the proposed extension.

Hypothesis: semantic top-k spends its whole budget on the frames that look
most like the question, and these often cluster on one moment. Spending part
of the same budget on frames temporally adjacent to the best matches may
capture context (approach and departure viewpoints, the surrounding
interaction) that the labels count as valid goal states.

Algorithm (budget k, radius r, anchor fraction f):

1. s_t = cos(e_q, e_t) for every valid frame (same scores as semantic top-k).
2. Anchors A: the top n_a = ceil(f * k) frames by s_t. These are exactly the
   first n_a picks of semantic top-k, so both methods share them.
3. Neighbour candidates N = {a + d : a in A, 1 <= |d| <= r} restricted to
   valid frames and excluding A. Overlapping neighbourhoods are deduplicated
   by construction (N is a set).
4. Neighbours are ranked by (distance to nearest anchor ascending,
   s_t descending, index ascending). Proximity comes first, so a frame
   adjacent to an anchor beats one two steps away; semantic relevance only
   breaks ties among equally close frames.
5. The remaining k - n_a slots are filled from N in that order.
6. If N runs out (small r, overlapping anchors, sequence edges), the
   remaining slots are backfilled with the next-best semantic frames.
7. Output is the k unique frames in chronological order.

Budget fairness: TNR and semantic top-k return exactly k frames. They agree
on the first n_a and differ only in how the remaining k - n_a slots are
spent: semantic ranks n_a+1..k versus temporal neighbours of the anchors.

r = 0 gives N = {}, so everything after the anchors is semantic backfill and
the output equals semantic top-k exactly.

The query embedding is a parameter, so the same code also runs with a fixed,
question-independent prompt (control condition).
"""

from __future__ import annotations

import math

import numpy as np

from src.memory.base import Retriever, stable_topk
from src.memory.frame_store import EpisodeMemory


class TemporalNeighborhoodRetriever(Retriever):
    name = "tnr"

    def __init__(self, radius: int = 2, anchor_frac: float = 0.5) -> None:
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if not 0 < anchor_frac <= 1:
            raise ValueError("anchor_frac must be in (0, 1]")
        self.radius = radius
        self.anchor_frac = anchor_frac

    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int):
        if query_emb is None:
            raise ValueError("TNR needs a query embedding")
        s = memory.similarity(query_emb)  # -inf on invalid frames
        T = len(s)
        n_anchor = min(k, math.ceil(self.anchor_frac * k))
        semantic_order = stable_topk(s, int(np.isfinite(s).sum()))  # full ranking of valid frames
        anchors = semantic_order[:n_anchor].tolist()
        chosen = list(anchors)
        taken = set(chosen)

        # neighbour candidates with their distance to the nearest anchor
        dist: dict[int, int] = {}
        for a in anchors:
            for d in range(1, self.radius + 1):
                for t in (a - d, a + d):
                    if 0 <= t < T and t not in taken and np.isfinite(s[t]):
                        dist[t] = min(dist.get(t, d), d)
        neighbours = sorted(dist, key=lambda t: (dist[t], -s[t], t))
        for t in neighbours:
            if len(chosen) == k:
                break
            chosen.append(t)
            taken.add(t)

        # backfill with next-best semantic frames
        for t in semantic_order:
            if len(chosen) == k:
                break
            if int(t) not in taken:
                chosen.append(int(t))
                taken.add(int(t))
        return chosen, [float(s[t]) for t in chosen]

    def config(self) -> dict:
        return {"radius": self.radius, "anchor_frac": self.anchor_frac}
