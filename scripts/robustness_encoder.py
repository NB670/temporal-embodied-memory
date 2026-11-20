"""Robustness check: repeat the core k=8 comparisons with a different encoder.

Usage: python -m scripts.robustness_encoder --model ViT-L-14 --pretrained laion2b_s32b_b82k
Writes results/tables/robustness_<model>_<pretrained>.csv
"""

from __future__ import annotations

import argparse
import copy

import pandas as pd

from src.data.load_dataset import PROJECT_ROOT
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions, run_grid
from src.evaluation.statistics import cluster_bootstrap, paired_difference
from src.memory.frame_store import FrameStore

GROUPS = {"Overall": None, "Spatial": "Single-Goal Spatial Tasks",
          "Temporal": "Single-Goal Temporal Tasks", "Multi-goal": "Multi-Goal Tasks"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    args = ap.parse_args()
    cfg = copy.deepcopy(load_config(PROJECT_ROOT / "configs" / "experiment.yaml"))
    cfg["encoder"].update(model_name=args.model, pretrained=args.pretrained)
    cfg.update(budgets=[4, 8, 16], radii=[0, 2], anchor_fracs=[0.5], random_seeds=[0])
    bs = dict(reps=cfg["statistics"]["bootstrap_reps"], seed=cfg["statistics"]["bootstrap_seed"])

    store = FrameStore(build_encoder(cfg))
    q = load_eval_questions(cfg)
    for r in q.drop_duplicates("ep_id").itertuples():
        store.get_episode(r.split, r.ep_id, r.video)
    raw = run_grid(cfg, store, q)
    raw = raw[(raw.method != "tnr") | (raw.radius == 2)]
    raw = raw[(raw.method != "tnr_generic") | (raw.radius == 2)]
    rows = []
    for k in cfg["budgets"]:
        for g, h in GROUPS.items():
            x = raw[(raw.k == k) & ((raw.high == h) if h else True)]
            w = x.pivot_table(index=["qid", "ep_id"], columns="method", values="goal_coverage").reset_index()
            for m in ["uniform", "semantic", "tnr", "semantic_generic", "tnr_generic"]:
                e = cluster_bootstrap(w[m].to_numpy(), w.ep_id.to_numpy(), **bs)
                rows.append(dict(encoder=store.encoder.config_id, k=k, group=g, stat=m, mean=e.mean, lo=e.lo, hi=e.hi))
            for a, b in [("tnr", "semantic"), ("semantic", "uniform"), ("semantic", "semantic_generic")]:
                e = paired_difference(w[a].to_numpy(), w[b].to_numpy(), w.ep_id.to_numpy(), **bs)
                rows.append(dict(encoder=store.encoder.config_id, k=k, group=g, stat=f"{a} - {b}",
                                 mean=e.mean, lo=e.lo, hi=e.hi, p=e.p_value))
    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "results" / "tables" / f"robustness_{args.model}_{args.pretrained}.csv"
    out.to_csv(path, index=False)
    print(out[out.k == 8].assign(**{c: lambda d, c=c: (d[c] * 100).round(1) for c in ["mean", "lo", "hi"]})
          .to_string(index=False))


if __name__ == "__main__":
    main()
