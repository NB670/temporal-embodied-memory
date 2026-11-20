"""Common retrieval interface shared by all frame-selection strategies.

Every retriever maps (episode memory, query embedding, budget k) to a
:class:`RetrievalResult`. Ground-truth labels are never an input: retrievers
see only the frame embeddings, the question embedding and the budget.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.memory.frame_store import EpisodeMemory


@dataclass(frozen=True)
class RetrievalResult:
    frame_indices: list[int]            # selected official indices, chronological order
    scores: list[float] | None          # per selected frame (same order), if applicable
    ranking: list[int]                  # selected indices in selection-priority order
    latency_ms: float                   # selection time only (embeddings precomputed)
    config: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    name: str = "base"

    def retrieve(self, memory: EpisodeMemory, query_emb: np.ndarray | None, k: int) -> RetrievalResult:
        """Validate inputs, time the selection, and check the output contract."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        valid_idx = np.flatnonzero(memory.valid)
        k_eff = min(k, len(valid_idx))
        t0 = time.perf_counter()
        ranking, scores = self._select(memory, query_emb, k_eff)
        latency = (time.perf_counter() - t0) * 1e3

        ranking = [int(i) for i in ranking]
        if len(ranking) != k_eff or len(set(ranking)) != k_eff:
            raise AssertionError(f"{self.name}: expected {k_eff} unique frames, got {ranking}")
        if not set(ranking) <= set(valid_idx.tolist()):
            raise AssertionError(f"{self.name}: returned invalid/out-of-range frames {ranking}")

        order = sorted(range(k_eff), key=lambda j: ranking[j])
        return RetrievalResult(
            frame_indices=[int(memory.frame_indices[ranking[j]]) for j in order],
            scores=None if scores is None else [float(scores[j]) for j in order],
            ranking=[int(memory.frame_indices[i]) for i in ranking],
            latency_ms=latency,
            config={"method": self.name, "k": k, "k_effective": k_eff, **self.config()},
        )

    @abstractmethod
    def _select(self, memory: EpisodeMemory, query_emb: np.ndarray | None,
                k: int) -> tuple[list[int], list[float] | None]:
        """Return (row positions in priority order, optional per-pick scores)."""

    def config(self) -> dict[str, Any]:
        return {}


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores; ties broken by lower index (deterministic)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]
