"""Pretrained image-text encoder (OpenCLIP) for frame and question embeddings.

Both modalities are L2-normalised, so the dot product equals cosine similarity:

    s_t = cos(E_text(q), E_image(I_t)) = <e_q, e_t>.

Preprocessing decisions (all part of :pyattr:`ClipEncoder.config_id`, which
keys the embedding cache):

* ``mask_overlay``: FindingDory frames have a burned-in overlay in the
  top-left ("Frame: N / Time of Day: HH:MM"). We black it out so similarity
  cannot come from rendered text that is unrelated to scene content.
* ``pad_to_square``: frames are portrait (640x480). CLIP's default resize plus
  center-crop would discard roughly 12% at the top and bottom. Padding keeps
  the whole field of view, at the cost of black borders.
* ``strip_prefix``: every question starts with "The robot's goal is: ".
  It carries no information and uses tokens of CLIP's 77-token context.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image

QUESTION_PREFIX = "The robot's goal is: "
# Overlay box in full-resolution (640x480) frame coordinates: rows, cols.
# Measured on decoded frames: text spans roughly y in [30, 80], x in [0, 170].
OVERLAY_BOX = (slice(0, 90), slice(0, 210))


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    mask_overlay: bool = True
    pad_to_square: bool = True
    strip_prefix: bool = True
    batch_size: int = 32


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_frame(frame: np.ndarray, cfg: EncoderConfig) -> Image.Image:
    """Apply overlay masking and square padding to an HxWx3 uint8 RGB frame."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 frame, got {frame.shape} {frame.dtype}")
    img = frame
    if cfg.mask_overlay:
        img = img.copy()
        img[OVERLAY_BOX] = 0
    if cfg.pad_to_square:
        h, w = img.shape[:2]
        s = max(h, w)
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        top, left = (s - h) // 2, (s - w) // 2
        canvas[top : top + h, left : left + w] = img
        img = canvas
    return Image.fromarray(img)


def clean_question(q: str, cfg: EncoderConfig) -> str:
    q = q.strip()
    if cfg.strip_prefix and q.startswith(QUESTION_PREFIX):
        q = q[len(QUESTION_PREFIX) :]
    return q


class ClipEncoder:
    """Thin wrapper around an OpenCLIP model returning normalised numpy arrays."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig(), device: str | None = None) -> None:
        import open_clip

        self.cfg = cfg
        self.device = device or pick_device()
        model, _, preprocess = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess  # resize(224) + center crop + normalise
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)
        self.open_clip_version = open_clip.__version__

    @property
    def config_id(self) -> str:
        """Stable short id of everything that affects embeddings."""
        blob = json.dumps({**asdict(self.cfg), "open_clip": self.open_clip_version}, sort_keys=True)
        tag = f"{self.cfg.model_name}_{self.cfg.pretrained}"
        return f"{tag}_{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

    @torch.no_grad()
    def encode_images(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return a ``[T, D]`` float32 array of L2-normalised image embeddings."""
        if len(frames) == 0:
            raise ValueError("encode_images got no frames")
        out = []
        for i in range(0, len(frames), self.cfg.batch_size):
            batch = torch.stack(
                [self.preprocess(preprocess_frame(f, self.cfg)) for f in frames[i : i + self.cfg.batch_size]]
            ).to(self.device)
            emb = self.model.encode_image(batch).float()
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return a ``[Q, D]`` float32 array of L2-normalised text embeddings."""
        if len(texts) == 0:
            raise ValueError("encode_texts got no texts")
        cleaned = [clean_question(t, self.cfg) for t in texts]
        out = []
        for i in range(0, len(cleaned), 256):
            tok = self.tokenizer(cleaned[i : i + 256]).to(self.device)
            emb = self.model.encode_text(tok).float()
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
