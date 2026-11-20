"""Phase 4/5 checkpoint: run every method on one episode/question and check outputs.

Usage: python -m scripts.run_baselines [--ep ep_2 --task task_41 --k 8]
Writes results/figures/phase5_semantic_vs_tnr_<ep>_<task>.png
"""

from __future__ import annotations

import argparse

from src.data.load_dataset import PROJECT_ROOT, RemoteZipVideoFetcher, decode_video
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions, method_grid
from src.evaluation.metrics import question_metrics
from src.memory.frame_store import FrameStore
from src.memory.semantic_retriever import SemanticRetriever
from src.memory.temporal_retriever import TemporalNeighborhoodRetriever
from src.visualization.frame_visualizer import plot_retrieval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--ep", default="ep_2")
    ap.add_argument("--task", default="task_41")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = FrameStore(build_encoder(cfg))
    q = load_eval_questions(cfg)
    r = q[(q.ep_id == args.ep) & (q.task_id == args.task)].iloc[0]
    mem = store.get_episode(r.split, r.ep_id, r.video)
    qe = store.text_embeddings([r.question])[0]
    ge = store.text_embeddings([cfg["generic_prompt"]])[0]
    print(f"{r.ep_id} {r.task_id} [{r.low_level_category}] {r.question[21:]}\nGT {r.answer}\n")
    seen = set()
    for name, ret, qtype in method_grid(cfg):
        key = (name, tuple(sorted(ret.config().items())))
        if name == "random" and ret.seed > 0 or key in seen:
            continue
        seen.add(key)
        emb = {"none": None, "question": qe, "generic": ge}[qtype]
        res = ret.retrieve(mem, emb, args.k)
        assert len(set(res.frame_indices)) == args.k and all(0 <= i < 96 for i in res.frame_indices)
        m = question_metrics(res.frame_indices, r.goals)
        print(f"{name:17s} {str(ret.config()):38s} {res.frame_indices}  "
              f"hit={m['any_hit']:.0f} prec={m['precision']:.2f}  {res.latency_ms:.3f}ms")

    sem = SemanticRetriever().retrieve(mem, qe, args.k)
    tnr = TemporalNeighborhoodRetriever(cfg["main_radius"], cfg["main_anchor_frac"]).retrieve(mem, qe, args.k)
    video = decode_video(RemoteZipVideoFetcher().fetch(r.video))
    out = plot_retrieval(video.frames, mem.similarity(qe),
                         {"semantic": sem.frame_indices, f"TNR r={cfg['main_radius']}": tnr.frame_indices},
                         r.goals, f"{r.ep_id} {r.task_id} [{r.low_level_category}]: {r.question[21:]}  (k={args.k})",
                         PROJECT_ROOT / "results" / "figures" / f"phase5_semantic_vs_tnr_{r.ep_id}_{r.task_id}.png",
                         scores={"semantic": sem.scores, f"TNR r={cfg['main_radius']}": tnr.scores})
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
