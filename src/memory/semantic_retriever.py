"""Baseline B: semantic top-k retrieval.

For question q and every valid frame t we compute s_t = <e_q, e_t> (cosine
similarity of unit-norm CLIP embeddings) and return the k distinct frames
with the highest s_t. Ties are broken by earlier frame index, so the result
is deterministic. Only the model's similarity scores are used, never labels.

Caveat: s_t measures how well frame content matches the question text. It
has no notion of order ("the object *after* the cup"), so for temporal
templates it can at best find frames showing the mentioned entities.
"""

from __future__ import annotations

import numpy as np

from src.memory.base import Retriever, stable_topk
from src.memory.frame_store import EpisodeMemory


class SemanticRetriever(Retriever):
    name = "semantic"

    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int):
        if query_emb is None:
            raise ValueError("SemanticRetriever needs a query embedding")
        s = memory.similarity(query_emb)  # invalid frames are -inf
        top = stable_topk(s, k)
        return top.tolist(), s[top].tolist()
