"""Loading utilities for the FindingDory 96-frame subsampled dataset.

Dataset: ``yali30/findingdory-subsampled-96`` (Apache-2.0), pinned to a fixed
revision for reproducibility.

Layout (verified 2026-11-20):
  * ``data/{train,validation}-00000-of-00001.parquet``: one row per question.
    ``video`` is a *relative path* such as ``videos/val/ep_1.mp4``, not bytes.
  * ``videos.zip`` (~3.19 GB): every mp4 for both splits.
  * ``episode_id_mappings.json``: original episode id -> renamed ``ep_<n>``.

``ep_id`` is only unique *within* a split (``ep_1`` exists in train and val and
refers to different videos), so we key episodes by ``(split, ep_id)``.

Instead of downloading the whole zip, :class:`RemoteZipVideoFetcher` reads the
zip central directory with HTTP Range requests and extracts single members.
Downloaded videos are cached under ``data/cache/videos/`` and never re-fetched.
"""

from __future__ import annotations

import ast
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

REPO_ID = "yali30/findingdory-subsampled-96"
REVISION = "95afe1e8ef355e06f27000851fc4033266bdd480"  # 2025-07-07 commit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
VIDEO_CACHE = PROJECT_ROOT / "data" / "cache" / "videos"
NUM_FRAMES = 96

Split = Literal["train", "validation"]
_PARQUET = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}

# Answer kinds, used to route questions to the right analysis.
ANSWER_FRAMES = "frames"        # every goal has >=1 valid frame index
ANSWER_UNSEEN = "unseen"        # [[-1]]: target never observed in the video
ANSWER_EMPTY = "empty"          # [[]]: no valid frame for the (only) goal
ANSWER_PARTIAL = "partial"      # multi-goal with some goals == [-1] or []


def parse_answer(answer: str) -> list[list[int]]:
    """Parse the stringified nested list into ``list[list[int]]``.

    Goal identity is preserved: one inner list per goal. Single-goal
    questions have exactly one inner list.
    """
    parsed = ast.literal_eval(answer)
    if not isinstance(parsed, list) or not all(isinstance(g, list) for g in parsed):
        raise ValueError(f"Unexpected answer structure: {answer!r}")
    for goal in parsed:
        for idx in goal:
            if not isinstance(idx, int) or not (-1 <= idx < NUM_FRAMES):
                raise ValueError(f"Out-of-range frame index {idx!r} in {answer!r}")
    return parsed


def classify_answer(goals: list[list[int]]) -> str:
    """Label an answer as frames / unseen / empty / partial (see constants)."""
    def bad(g: list[int]) -> bool:
        return len(g) == 0 or g == [-1]

    if all(not bad(g) for g in goals):
        return ANSWER_FRAMES
    if len(goals) == 1:
        return ANSWER_UNSEEN if goals[0] == [-1] else ANSWER_EMPTY
    return ANSWER_PARTIAL


def load_questions(split: Split = "validation") -> pd.DataFrame:
    """Load question metadata for a split, with parsed answers.

    Adds columns: ``split``, ``goals`` (parsed answer), ``num_goals``,
    ``answer_kind``.
    """
    path = hf_hub_download(
        REPO_ID, _PARQUET[split], repo_type="dataset", revision=REVISION, local_dir=RAW_DIR
    )
    df = pd.read_parquet(path)
    df["split"] = split
    df["goals"] = df["answer"].map(parse_answer)
    df["num_goals"] = df["goals"].map(len)
    df["answer_kind"] = df["goals"].map(classify_answer)
    return df


def select_episodes(df: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """Choose ``n`` episode ids, stratified by ``num_interactions``.

    Selection uses only episode metadata, never model outputs, so the subset
    cannot be biased toward episodes where a method happens to work well.
    """
    eps = df.drop_duplicates("ep_id")[["ep_id", "num_interactions"]]
    strata = sorted(eps["num_interactions"].unique())
    per = max(1, n // len(strata))
    chosen: list[str] = []
    for s in strata:
        group = eps[eps["num_interactions"] == s].sort_values("ep_id")
        chosen += group.sample(n=min(per, len(group)), random_state=seed)["ep_id"].tolist()
    if len(chosen) < n:  # top up deterministically if n is not divisible
        rest = eps[~eps["ep_id"].isin(chosen)].sort_values("ep_id")
        chosen += rest.sample(n=n - len(chosen), random_state=seed)["ep_id"].tolist()
    return sorted(chosen[:n], key=lambda e: int(e.split("_")[1]))


class _HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP Range requests (enough for zipfile)."""

    def __init__(self, url: str, session: requests.Session) -> None:
        self.session = session
        head = session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.url = head.url  # resolved CDN URL
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        if r.status_code != 206:
            raise IOError(f"Range request failed with HTTP {r.status_code}")
        data = r.content
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data

    def readinto(self, b) -> int:  # used by io.BufferedReader
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


class RemoteZipVideoFetcher:
    """Fetch individual mp4s from the remote ``videos.zip`` and cache them."""

    def __init__(self, cache_dir: Path = VIDEO_CACHE) -> None:
        self.cache_dir = Path(cache_dir)
        self._zip: zipfile.ZipFile | None = None
        self._raw: _HttpRangeFile | None = None

    def _open(self) -> zipfile.ZipFile:
        if self._zip is None:
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/videos.zip"
            self._raw = _HttpRangeFile(url, requests.Session())
            # Large buffer -> few HTTP requests per member.
            self._zip = zipfile.ZipFile(io.BufferedReader(self._raw, buffer_size=1 << 20))
        return self._zip

    def list_members(self) -> list[str]:
        return [n for n in self._open().namelist() if n.endswith(".mp4")]

    def local_path(self, rel_video: str) -> Path:
        return self.cache_dir / rel_video

    def fetch(self, rel_video: str) -> Path:
        """Return a local path for ``rel_video`` (e.g. ``videos/val/ep_1.mp4``)."""
        out = self.local_path(rel_video)
        if out.exists() and out.stat().st_size > 0:
            return out
        zf = self._open()
        info = zf.getinfo(rel_video)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        with zf.open(info) as src, open(tmp, "wb") as dst:
            dst.write(src.read())  # zipfile validates CRC on read
        tmp.rename(out)
        return out

    @property
    def bytes_fetched(self) -> int:
        return self._raw.bytes_fetched if self._raw else 0


@dataclass(frozen=True)
class DecodedVideo:
    frames: "list"  # list of HxWx3 uint8 RGB arrays, chronological
    fps: float
    frame_indices: list[int]  # official dataset index for each decoded frame


def decode_video(path: Path) -> DecodedVideo:
    """Decode all frames in order. Frame ``i`` of the decoded stream is taken
    to be dataset frame index ``i``; :func:`inspect_dataset` verifies 96 frames.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return DecodedVideo(frames=frames, fps=fps, frame_indices=list(range(len(frames))))
