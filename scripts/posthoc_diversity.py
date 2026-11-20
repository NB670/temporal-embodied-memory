"""POST-HOC exploratory analysis: temporal diversity (semantic + NMS) vs semantic top-k.

Added after the pre-specified results showed TNR's neighbours are redundant.
Results are written separately (results/tables/posthoc_diversity.csv) and must
be reported as post-hoc, not as a confirmatory test.

Usage: python -m scripts.posthoc_diversity
"""

from __future__ import annotations

import pandas as pd

from src.data.load_dataset import ANSWER_FRAMES, PROJECT_ROOT
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions
from src.evaluation.metrics import question_metrics
from src.evaluation.statistics import cluster_bootstrap, paired_difference
from src.memory.diversity_retriever import SemanticNMSRetriever
from src.memory.frame_store import FrameStore
from src.memory.semantic_retriever import SemanticRetriever
from src.memory.uniform_retriever import UniformRetriever

TAB = PROJECT_ROOT / "results" / "tables"
GROUPS = {"Overall": None, "Spatial": "Single-Goal Spatial Tasks",
          "Temporal": "Single-Goal Temporal Tasks", "Multi-goal": "Multi-Goal Tasks"}


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    bs = dict(reps=cfg["statistics"]["bootstrap_reps"], seed=cfg["statistics"]["bootstrap_seed"])
    store = FrameStore(build_encoder(cfg))
    q = load_eval_questions(cfg)
    q = q[q.answer_kind == ANSWER_FRAMES]
    qe = store.text_embeddings(q.question.tolist())
    ge = store.text_embeddings([cfg["generic_prompt"]])[0]
    methods = [("semantic", SemanticRetriever(), "q"), ("uniform", UniformRetriever(), "q"),
               ("semantic_generic", SemanticRetriever(), "g")]
    methods += [(f"nms_g{g}", SemanticNMSRetriever(g), "q") for g in (1, 2, 4, 8)]
    methods += [(f"nms_generic_g{g}", SemanticNMSRetriever(g), "g") for g in (1, 2, 4, 8)]
    rows = []
    for j, r in enumerate(q.itertuples()):
        mem = store.get_episode(r.split, r.ep_id, r.video)
        for name, ret, t in methods:
            for k in cfg["budgets"]:
                res = ret.retrieve(mem, qe[j] if t == "q" else ge, k)
                rows.append(dict(qid=r.qid, ep_id=r.ep_id, high=r.high_level_category, k=k, method=name,
                                 **question_metrics(res.frame_indices, r.goals)))
    d = pd.DataFrame(rows)
    out = []
    for k in cfg["budgets"]:
        for g, h in GROUPS.items():
            x = d[(d.k == k) & ((d.high == h) if h else True)]
            w = x.pivot_table(index=["qid", "ep_id"], columns="method", values="goal_coverage").reset_index()
            for m in [c for c in w.columns if c not in ("qid", "ep_id")]:
                lvl = cluster_bootstrap(w[m].to_numpy(), w.ep_id.to_numpy(), **bs)
                base = "semantic_generic" if "generic" in m else "semantic"
                dif = paired_difference(w[m].to_numpy(), w[base].to_numpy(), w.ep_id.to_numpy(), **bs)
                out.append(dict(k=k, group=g, method=m, coverage=lvl.mean, lo=lvl.lo, hi=lvl.hi,
                                diff_vs=base, diff=dif.mean, diff_lo=dif.lo, diff_hi=dif.hi, p=dif.p_value,
                                n_questions=lvl.n_questions))
    o = pd.DataFrame(out)
    o.to_csv(TAB / "posthoc_diversity.csv", index=False)
    show = o[(o.group == "Overall")].pivot_table(index="method", columns="k", values="coverage").mul(100).round(1)
    print("POST-HOC: overall goal coverage (%)\n", show.to_string())
    s8 = o[(o.k == 8)][["group", "method", "coverage", "diff", "diff_lo", "diff_hi", "p"]]
    print("\nk=8 differences vs semantic (or vs semantic_generic for generic variants), pp:")
    print(s8.assign(**{c: s8[c] * 100 for c in ["coverage", "diff", "diff_lo", "diff_hi"]}).round(1)
          .pivot_table(index="method", columns="group", values="diff").to_string())


if __name__ == "__main__":
    main()
