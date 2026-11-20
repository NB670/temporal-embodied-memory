"""Per-episode visual memory: frame embeddings plus their official indices.

An :class:`EpisodeMemory` is the memory ``V = {I_1..I_T}`` after encoding:

* ``embeddings[t]``: L2-normalised image embedding of decoded frame ``t``
* ``frame_indices[t]``: official dataset index of that frame (0..95)
* ``timestamps[t]``: seconds from episode start (``t / fps``; the 96-frame
  videos are stored at 1 fps, so this is the position in the subsampled video,
  not the real elapsed time. That time appears only in the overlay.)
* ``valid[t]``: False if the frame was missing or undecodable. Retrievers
  must never return invalid frames.

Embeddings are cached as ``.npz`` under
``data/cache/embeddings/<encoder.config_id>/<split>__<ep_id>.npz``, so each
episode is decoded and encoded once per encoder configuration. Text
embeddings are cached per question string in one ``.npz`` per config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.data.load_dataset import NUM_FRAMES, PROJECT_ROOT, RemoteZipVideoFetcher, decode_video

EMB_CACHE = PROJECT_ROOT / "data" / "cache" / "embeddings"


class Encoder(Protocol):
    config_id: str

    def encode_images(self, frames: list[np.ndarray]) -> np.ndarray: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeMemory:
    split: str
    ep_id: str
    embeddings: np.ndarray      # [T, D] float32, unit norm on valid rows
    frame_indices: np.ndarray   # [T] int
    timestamps: np.ndarray      # [T] float seconds
    valid: np.ndarray           # [T] bool

    def __post_init__(self) -> None:
        T = len(self.frame_indices)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != T:
            raise ValueError(f"embeddings {self.embeddings.shape} vs {T} frames")
        if self.timestamps.shape != (T,) or self.valid.shape != (T,):
            raise ValueError("timestamps/valid must be shape [T]")
        if T and not np.all(np.diff(self.frame_indices) > 0):
            raise ValueError("frame_indices must be strictly increasing (chronological)")

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def similarity(self, text_emb: np.ndarray) -> np.ndarray:
        """Cosine similarity of every frame to a unit-norm text embedding.

        Invalid frames get ``-inf`` so they can never be ranked.
        """
        if text_emb.ndim != 1 or text_emb.shape[0] != self.embeddings.shape[1]:
            raise ValueError(f"text_emb shape {text_emb.shape} vs dim {self.embeddings.shape[1]}")
        s = self.embeddings @ text_emb
        return np.where(self.valid, s, -np.inf).astype(np.float64)


class FrameStore:
    """Builds and caches :class:`EpisodeMemory` objects for one encoder."""

    def __init__(self, encoder: Encoder, cache_root: Path = EMB_CACHE,
                 fetcher: RemoteZipVideoFetcher | None = None) -> None:
        self.encoder = encoder
        self.cache_dir = Path(cache_root) / encoder.config_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or RemoteZipVideoFetcher()
        self._text_path = self.cache_dir / "text_embeddings.npz"
        self._text: dict[str, np.ndarray] = {}
        if self._text_path.exists():
            z = np.load(self._text_path, allow_pickle=False)
            self._text = dict(zip(z["texts"].tolist(), z["emb"]))

    def _ep_path(self, split: str, ep_id: str) -> Path:
        return self.cache_dir / f"{split}__{ep_id}.npz"

    def get_episode(self, split: str, ep_id: str, rel_video: str,
                    expected_frames: int = NUM_FRAMES) -> EpisodeMemory:
        path = self._ep_path(split, ep_id)
        if path.exists():
            z = np.load(path, allow_pickle=False)
            return EpisodeMemory(split, ep_id, z["embeddings"], z["frame_indices"], z["timestamps"], z["valid"])

        video = decode_video(self.fetcher.fetch(rel_video))
        mem = build_memory(split, ep_id, video.frames, video.fps, self.encoder, expected_frames)
        np.savez(path, embeddings=mem.embeddings, frame_indices=mem.frame_indices,
                 timestamps=mem.timestamps, valid=mem.valid)
        meta = dict(split=split, ep_id=ep_id, video=rel_video, encoder=self.encoder.config_id,
                    decoded_frames=len(video.frames), n_valid=int(mem.valid.sum()), fps=video.fps)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        return mem

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        """Embeddings for ``texts`` in order, computing and caching misses."""
        missing = sorted({t for t in texts if t not in self._text})
        if missing:
            for t, e in zip(missing, self.encoder.encode_texts(missing)):
                self._text[t] = e
            keys = list(self._text)
            np.savez(self._text_path, texts=np.array(keys), emb=np.stack([self._text[k] for k in keys]))
        return np.stack([self._text[t] for t in texts])


def build_memory(split: str, ep_id: str, frames: list[np.ndarray | None], fps: float,
                 encoder: Encoder, expected_frames: int = NUM_FRAMES) -> EpisodeMemory:
    """Encode decoded frames into an :class:`EpisodeMemory`.

    Missing or corrupted frames (``None``, wrong shape, all-zero, or frames
    absent because decoding stopped early) are kept as rows with
    ``valid=False`` and zero embeddings. Indices therefore stay aligned with
    the dataset's 0..expected_frames-1 labels.
    """
    T = max(expected_frames, len(frames))
    padded = list(frames) + [None] * (T - len(frames))
    ok = [f is not None and f.ndim == 3 and f.shape[2] == 3 and f.any() for f in padded]
    good = [f for f, o in zip(padded, ok) if o]
    if not good:
        raise ValueError(f"{split}/{ep_id}: no decodable frames")
    emb_good = encoder.encode_images(good)
    emb = np.zeros((T, emb_good.shape[1]), dtype=np.float32)
    emb[np.array(ok)] = emb_good
    fps = fps if fps and fps > 0 else 1.0
    idx = np.arange(T)
    return EpisodeMemory(split, ep_id, emb, idx, idx / fps, np.array(ok))
