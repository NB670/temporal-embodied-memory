"""Encode every evaluation episode once and cache frame + question embeddings.

Usage: python -m scripts.prepare_data [--config configs/experiment.yaml]
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from src.data.load_dataset import PROJECT_ROOT
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions
from src.memory.frame_store import FrameStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(0)
    enc = build_encoder(cfg)
    store = FrameStore(enc)
    q = load_eval_questions(cfg)
    eps = q.drop_duplicates("ep_id")[["split", "ep_id", "video"]]

    t0 = time.perf_counter()
    n_invalid = 0
    for r in eps.itertuples():
        mem = store.get_episode(r.split, r.ep_id, r.video)
        if not mem.valid.all():
            n_invalid += int((~mem.valid).sum())
            print(f"WARNING {r.ep_id}: {int((~mem.valid).sum())} invalid frames")
    t_img = time.perf_counter() - t0
    t0 = time.perf_counter()
    store.text_embeddings(q.question.tolist() + [cfg["generic_prompt"]])
    t_txt = time.perf_counter() - t0

    env = dict(encoder=enc.config_id, device=enc.device, torch=torch.__version__,
               open_clip=enc.open_clip_version, python=platform.python_version(),
               machine=platform.machine(), platform=platform.platform(),
               n_episodes=len(eps), n_questions=len(q), n_unique_questions=q.question.nunique(),
               n_invalid_frames=n_invalid,
               image_encode_s_uncached_part=round(t_img, 2), text_encode_s=round(t_txt, 2))
    (store.cache_dir / "environment.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
