import numpy as np
import pytest

from src.memory.frame_store import EpisodeMemory


class FakeEncoder:
    """Deterministic encoder: embedding = normalised per-channel means + noise by content."""
    config_id = "fake_v1"

    def __init__(self):
        self.image_calls = 0
        self.text_calls = 0

    def encode_images(self, frames):
        self.image_calls += 1
        e = np.stack([np.r_[f.reshape(-1, 3).mean(0), 1.0] for f in frames]).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def encode_texts(self, texts):
        self.text_calls += 1
        e = np.stack([np.r_[len(t), t.count("a"), t.count("e"), 1.0] for t in texts]).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


def make_memory(scores, valid=None, dim=2):
    """Memory whose similarity to query e_0 equals ``scores`` (approximately, via angle)."""
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    ang = np.arccos(np.clip(scores, -1, 1))
    emb = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    valid = np.ones(T, bool) if valid is None else np.asarray(valid)
    emb[~valid] = 0
    idx = np.arange(T)
    return EpisodeMemory("validation", "ep_test", emb, idx, idx.astype(float), valid), np.array([1.0, 0.0], np.float32)
