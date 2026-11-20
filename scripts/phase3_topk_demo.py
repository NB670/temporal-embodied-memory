"""Phase 3 checkpoint: semantic top-k for fixed example questions, plus a
signal diagnostic: does CLIP similarity rank GT frames above non-GT frames?

Example selection rule (fixed in advance, no cherry-picking): first subset
episode; first question of each of three low-level categories.

Diagnostic: for each question with answer_kind == frames, compute AUROC of the
similarity scores with GT-union frames as positives (0.5 = no signal). This
uses labels for *evaluation* only.

Usage: python -m scripts.phase3_topk_demo [--k 8]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.load_dataset import PROJECT_ROOT, RemoteZipVideoFetcher, decode_video
from src.memory.frame_store import FrameStore
from src.memory.semantic_retriever import SemanticRetriever
from src.models.encoder import ClipEncoder
from src.visualization.frame_visualizer import plot_retrieval

META = PROJECT_ROOT / "data" / "metadata"
FIG = PROJECT_ROOT / "results" / "figures"
TAB = PROJECT_ROOT / "results" / "tables"
DEMO_CATEGORIES = ["Object Recall", "Interaction Order", "Unordered Revisitation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    enc = ClipEncoder()
    store = FrameStore(enc)
    q = pd.read_csv(META / "subset_questions.csv")
    q["goals"] = q["goals"].map(json.loads)
    q["emb_row"] = range(len(q))
    text = store.text_embeddings(q.question.tolist())
    retriever = SemanticRetriever()

    # ---- examples --------------------------------------------------------------------
    ep0 = json.loads((META / "subset_episodes.json").read_text())["ep_ids"][0]
    sub = q[(q.ep_id == ep0) & (q.answer_kind == "frames")]
    ex = sub[sub.low_level_category.isin(DEMO_CATEGORIES)].drop_duplicates("low_level_category")
    video = decode_video(RemoteZipVideoFetcher().fetch(ex.video.iloc[0]))
    for r in ex.itertuples():
        mem = store.get_episode(r.split, r.ep_id, r.video)
        res = retriever.retrieve(mem, text[r.emb_row], args.k)
        gt = {i for g in r.goals for i in g}
        print(f"\n[{r.ep_id} {r.task_id} | {r.low_level_category}] {r.question[21:]}")
        print(f"  GT: {r.answer}")
        print(f"  top-{args.k} by score: {res.ranking}")
        print(f"  scores: {[round(s, 3) for s in sorted(res.scores, reverse=True)]}")
        print(f"  chronological: {res.frame_indices}  hits: {sorted(gt & set(res.frame_indices))}"
              f"  latency {res.latency_ms:.3f} ms")
        out = plot_retrieval(video.frames, mem.similarity(text[r.emb_row]),
                             {"semantic": res.frame_indices}, r.goals,
                             f"{r.ep_id} {r.task_id} [{r.low_level_category}]: {r.question[21:]}",
                             FIG / f"phase3_topk_{r.task_id}.png", scores={"semantic": res.scores})
        print(f"  figure: {out}")

    # ---- signal diagnostic -----------------------------------------------------------
    # Control: score the same GT with a *mismatched* question from the same episode
    # (deterministic: the question half an episode away in row order). If AUROC stays
    # high, the signal is a generic "interaction-frame" prior, not question-specific.
    rows = []
    for ep, g in q.groupby("ep_id", sort=False):
        rows_ep = g.emb_row.tolist()
        mem = store.get_episode(g.split.iloc[0], ep, g.video.iloc[0])
        for j, r in enumerate(g.itertuples()):
            if r.answer_kind != "frames":
                continue
            y = np.zeros(mem.num_frames, bool)
            y[[i for gg in r.goals for i in gg]] = True
            if not 0 < y.sum() < len(y):
                continue
            other = rows_ep[(j + len(rows_ep) // 2) % len(rows_ep)]
            rows.append(dict(ep_id=ep, high=r.high_level_category, low=r.low_level_category,
                             auroc=roc_auc_score(y, mem.similarity(text[r.emb_row])),
                             auroc_mismatched=roc_auc_score(y, mem.similarity(text[other]))))
    d = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    d.to_csv(TAB / "phase3_similarity_auroc.csv", index=False)
    # episode-level averaging first, so episodes (not questions) are the unit
    ep_means = d.groupby(["high", "low", "ep_id"])[["auroc", "auroc_mismatched"]].mean()
    by = ep_means.groupby(["high", "low"]).agg(["mean", "std"])
    by.columns = ["auroc", "auroc_sd", "mismatched", "mismatched_sd"]
    by["gap"] = by.auroc - by.mismatched
    by["n_eps"] = ep_means.groupby(["high", "low"]).size()
    by["n_q"] = d.groupby(["high", "low"]).size()
    print("\nAUROC of cos(question, frame), GT vs non-GT frames (0.5 = no signal), mean over episodes;")
    print("'mismatched' = same GT scored with another question of the same episode:")
    print(by.round(3).to_string())
    e = d.groupby("ep_id")[["auroc", "auroc_mismatched"]].mean()
    print(f"\noverall: matched {e.auroc.mean():.3f}  mismatched {e.auroc_mismatched.mean():.3f}"
          f"  (n_q={len(d)}, n_eps={len(e)})")

if __name__ == "__main__":
    main()
