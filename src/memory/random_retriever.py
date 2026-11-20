"""Baseline C: uniformly random sampling without replacement.

The per-call random stream is derived from (seed, split, episode, query bytes),
so a draw is reproducible and independent of evaluation order. Different
seeds give independent replicates; evaluation averages over seeds and
reports their spread.
"""

from __future__ import annotations

import zlib

import numpy as np

from src.memory.base import Retriever
from src.memory.frame_store import EpisodeMemory


class RandomRetriever(Retriever):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int):
        key = f"{memory.split}/{memory.ep_id}".encode()
        if query_emb is not None:
            key += np.ascontiguousarray(query_emb).tobytes()
        rng = np.random.default_rng([self.seed, zlib.crc32(key)])
        valid = np.flatnonzero(memory.valid)
        return rng.choice(valid, size=k, replace=False).tolist(), None

    def config(self) -> dict:
        return {"seed": self.seed}
