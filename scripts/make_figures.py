"""Phase 7 (Experiment 5) + Phase 9: qualitative cases and all research figures.

Qualitative selection rule (fixed, not cherry-picked): single-goal questions at
k=8, split into the four cells of (semantic hit?, TNR hit?); draw 2 questions
per cell with numpy default_rng(0), from different episodes.

Usage: python -m scripts.make_figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_dataset import PROJECT_ROOT, RemoteZipVideoFetcher, decode_video
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions
from src.memory.frame_store import FrameStore
from src.visualization import plots
from src.visualization.frame_visualizer import plot_retrieval

TAB = PROJECT_ROOT / "results" / "tables"
FIG = PROJECT_ROOT / "results" / "figures"
MAIN = ["random", "uniform", "semantic", "tnr", "semantic_generic", "tnr_generic"]


def n_segments(sel: str) -> int:
    a = sorted(map(int, sel.split()))
    return 1 + sum(b - x > 1 for x, b in zip(a, a[1:]))


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    e1 = pd.read_csv(TAB / "e1_methods_k8.csv")
    e2 = pd.read_csv(TAB / "e2_budget_methods.csv")
    e3 = pd.read_csv(TAB / "e3_radius_anchor.csv")
    e4 = pd.read_csv(TAB / "e4_lowlevel_k8.csv")
    ph = pd.read_csv(TAB / "posthoc_diversity.csv")
    raw = pd.read_parquet(TAB / "raw_results.parquet")

    print(plots.method_comparison(e1, MAIN, FIG / "fig1_method_comparison_k8.png"))
    print(plots.budget_curves(e2, ["random", "uniform", "semantic", "tnr", "semantic_generic"],
                              FIG / "fig2_coverage_vs_budget.png"))
    print(plots.radius_ablation(e3, FIG / "fig3_radius_ablation.png"))
    print(plots.category_forest(e4, FIG / "fig4_category_differences.png"))

    r, af = cfg["main_radius"], cfg["main_anchor_frac"]
    main = raw[raw.method.isin(["uniform", "semantic", "semantic_generic"])
               | ((raw.method == "tnr") & (raw.radius == r) & (raw.anchor_frac == af))
               | ((raw.method == "random") & (raw.seed == 0))]
    m8 = main[main.k == 8].assign(n_segments=lambda x: x.selected.map(n_segments))
    seg = m8.groupby("method", as_index=False).n_segments.mean()
    p = main[main.method.isin(["semantic", "tnr"])].pivot_table(index="qid", columns=["method", "k"],
                                                                values="goal_coverage")
    a4 = p[("semantic", 4)]
    marg = pd.DataFrame({"method": ["semantic", "tnr"],
                         "gain": [(p[("semantic", 8)] - a4).mean(), (p[("tnr", 8)] - a4).mean()]})
    pd.concat([seg.assign(stat="n_segments_k8"), marg.assign(stat="gain_slots_5_8")]).to_csv(
        TAB / "mechanism.csv", index=False)
    print(plots.mechanism(seg, marg, FIG / "fig5_mechanism.png"))
    print(plots.posthoc_diversity(ph, e2, FIG / "fig6_posthoc_diversity.png"))

    # ---------------- Experiment 5: qualitative cells ---------------------------------
    sg = m8[(m8.num_goals == 1) & m8.method.isin(["semantic", "tnr"])]
    w = sg.pivot_table(index=["qid", "ep_id"], columns="method", values="any_hit").reset_index()
    cells = {"semantic_fails_tnr_succeeds": (0, 1), "tnr_fails_semantic_succeeds": (1, 0),
             "both_fail": (0, 0), "both_succeed": (1, 1)}
    counts, picks = [], []
    rng = np.random.default_rng(0)
    for name, (s, t) in cells.items():
        c = w[(w.semantic == s) & (w.tnr == t)]
        counts.append(dict(cell=name, n=len(c), frac=len(c) / len(w)))
        c = c.sample(frac=1, random_state=int(rng.integers(1 << 31))).drop_duplicates("ep_id").head(2)
        picks += [(name, q) for q in c.qid]
    pd.DataFrame(counts).to_csv(TAB / "e5_outcome_cells_k8.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False))

    store = FrameStore(build_encoder(cfg))
    from src.memory.semantic_retriever import SemanticRetriever
    from src.memory.temporal_retriever import TemporalNeighborhoodRetriever

    qs = load_eval_questions(cfg).set_index("qid")
    fetch = RemoteZipVideoFetcher()
    rows = []
    for i, (cell, qid) in enumerate(picks):
        q = qs.loc[qid]
        mem = store.get_episode(q.split, q.ep_id, q.video)
        e = store.text_embeddings([q.question])[0]
        sem = SemanticRetriever().retrieve(mem, e, 8)
        tnr = TemporalNeighborhoodRetriever(r, af).retrieve(mem, e, 8)
        frames = decode_video(fetch.fetch(q.video)).frames
        out = plot_retrieval(frames, mem.similarity(e), {"semantic": sem.frame_indices, "TNR": tnr.frame_indices},
                             q.goals, f"[{cell}] {q.ep_id} {q.task_id} [{q.low_level_category}]: {q.question[21:]}",
                             FIG / "qualitative" / f"e5_{cell}_{i % 2}.png",
                             scores={"semantic": sem.scores, "TNR": tnr.scores})
        rows.append(dict(cell=cell, qid=qid, low=q.low_level_category, question=q.question[21:],
                         gt=q.answer, semantic=sem.frame_indices, tnr=tnr.frame_indices, figure=out.name))
    pd.DataFrame(rows).to_csv(TAB / "e5_qualitative_examples.csv", index=False)
    print(pd.DataFrame(rows)[["cell", "qid", "low", "gt", "semantic", "tnr"]].to_string(index=False))


if __name__ == "__main__":
    main()
