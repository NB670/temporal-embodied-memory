"""Run every retrieval method over every evaluable question and score it.

The output is a long table, one row per (question, method, k, radius,
anchor_frac, seed), with the selected frames, metrics and selection latency.
Retrievers receive only the episode memory, a query embedding and k. Ground
truth is used afterwards, in :func:`question_metrics`, never before.

Method names
  uniform            Baseline A (query-agnostic, deterministic)
  random             Baseline C (one row per seed)
  semantic           Baseline B with the question embedding
  tnr                TNR with the question embedding
  semantic_generic   control: semantic top-k with a fixed generic prompt
  tnr_generic        control: TNR with the same generic prompt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.load_dataset import ANSWER_FRAMES, PROJECT_ROOT, load_questions
from src.evaluation.metrics import question_metrics
from src.memory.frame_store import FrameStore
from src.memory.random_retriever import RandomRetriever
from src.memory.semantic_retriever import SemanticRetriever
from src.memory.temporal_retriever import TemporalNeighborhoodRetriever
from src.memory.uniform_retriever import UniformRetriever
from src.models.encoder import ClipEncoder, EncoderConfig


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def build_encoder(cfg: dict) -> ClipEncoder:
    return ClipEncoder(EncoderConfig(**cfg["encoder"]))


def load_eval_questions(cfg: dict) -> pd.DataFrame:
    """All questions of the configured episodes (every answer kind)."""
    d = cfg["dataset"]
    q = load_questions(d["split"])
    if d["episodes"] != "all":
        eps = json.loads((PROJECT_ROOT / d["episodes"]).read_text())["ep_ids"]
        q = q[q.ep_id.isin(eps)]
    dev = set(json.loads((PROJECT_ROOT / d["dev_subset"]).read_text())["ep_ids"])
    q = q.assign(dev_subset=q.ep_id.isin(dev)).reset_index(drop=True)
    q["qid"] = q.split + "/" + q.ep_id + "/" + q.task_id
    return q


def method_grid(cfg: dict) -> list[tuple[str, Any, str]]:
    """(method name, retriever, query type) for every configuration."""
    grid: list[tuple[str, Any, str]] = [("uniform", UniformRetriever(), "none")]
    grid += [("random", RandomRetriever(s), "question") for s in cfg["random_seeds"]]
    grid += [("semantic", SemanticRetriever(), "question"),
             ("semantic_generic", SemanticRetriever(), "generic")]
    for r in cfg["radii"]:
        for af in cfg["anchor_fracs"]:
            if r == 0 and af != cfg["main_anchor_frac"]:
                continue  # r=0 is identical to semantic for every anchor_frac
            grid.append(("tnr", TemporalNeighborhoodRetriever(r, af), "question"))
        grid.append(("tnr_generic", TemporalNeighborhoodRetriever(r, cfg["main_anchor_frac"]), "generic"))
    return grid


def run_grid(cfg: dict, store: FrameStore, questions: pd.DataFrame) -> pd.DataFrame:
    q = questions[questions.answer_kind == ANSWER_FRAMES]
    q_emb = store.text_embeddings(q.question.tolist())
    g_emb = store.text_embeddings([cfg["generic_prompt"]])[0]
    grid = method_grid(cfg)
    rows = []
    for j, r in enumerate(q.itertuples()):
        mem = store.get_episode(r.split, r.ep_id, r.video)
        for name, ret, qtype in grid:
            emb = {"none": None, "question": q_emb[j], "generic": g_emb}[qtype]
            for k in cfg["budgets"]:
                res = ret.retrieve(mem, emb, k)
                m = question_metrics(res.frame_indices, r.goals)
                rows.append(dict(
                    qid=r.qid, ep_id=r.ep_id, task_id=r.task_id, high=r.high_level_category,
                    low=r.low_level_category, num_interactions=r.num_interactions,
                    num_goals=r.num_goals, dev_subset=r.dev_subset, method=name, k=k,
                    radius=res.config.get("radius", np.nan), anchor_frac=res.config.get("anchor_frac", np.nan),
                    seed=res.config.get("seed", np.nan), latency_ms=res.latency_ms,
                    selected=" ".join(map(str, res.frame_indices)), **m))
    return pd.DataFrame(rows)
