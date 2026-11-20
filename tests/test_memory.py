import numpy as np
import pytest

from src.memory.frame_store import EpisodeMemory, FrameStore, build_memory
from src.memory.semantic_retriever import SemanticRetriever
from src.models.encoder import EncoderConfig, clean_question, preprocess_frame
from tests.conftest import make_memory


def frames(n, h=8, w=6):
    rng = np.random.default_rng(0)
    return [rng.integers(1, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_preprocess_masks_overlay_and_pads():
    f = np.full((640, 480, 3), 200, np.uint8)
    img = np.asarray(preprocess_frame(f, EncoderConfig()))
    assert img.shape == (640, 640, 3)
    assert img[:, :80].max() == 0                  # left pad
    assert img[10, 80 + 10].max() == 0             # overlay masked (offset by pad)
    assert img[320, 320].min() == 200              # content kept


def test_clean_question():
    assert clean_question("The robot's goal is: Navigate to a cup.", EncoderConfig()) == "Navigate to a cup."


def test_build_memory_marks_missing_and_short(fake_encoder):
    fr = frames(5)
    fr[2] = None
    fr[3] = np.zeros_like(fr[0])                   # all-black = corrupted
    mem = build_memory("validation", "ep_x", fr, 1.0, fake_encoder, expected_frames=8)
    assert mem.num_frames == 8
    assert mem.valid.tolist() == [True, True, False, False, True, False, False, False]
    assert np.allclose(np.linalg.norm(mem.embeddings[mem.valid], axis=1), 1)
    assert np.all(mem.embeddings[~mem.valid] == 0)


def test_memory_rejects_non_chronological():
    with pytest.raises(ValueError):
        EpisodeMemory("s", "e", np.zeros((3, 2), np.float32), np.array([0, 2, 1]), np.zeros(3), np.ones(3, bool))


def test_frame_store_caches(tmp_path, fake_encoder, monkeypatch):
    import src.memory.frame_store as fs
    from src.data.load_dataset import DecodedVideo

    calls = {"n": 0}

    def fake_decode(_):
        calls["n"] += 1
        return DecodedVideo(frames(6), 1.0, list(range(6)))

    class FakeFetcher:
        def fetch(self, rel):
            return tmp_path / rel

    monkeypatch.setattr(fs, "decode_video", fake_decode)
    store = FrameStore(fake_encoder, cache_root=tmp_path, fetcher=FakeFetcher())
    m1 = store.get_episode("validation", "ep_1", "v.mp4", expected_frames=6)
    m2 = FrameStore(fake_encoder, cache_root=tmp_path, fetcher=FakeFetcher()).get_episode(
        "validation", "ep_1", "v.mp4", expected_frames=6)
    assert calls["n"] == 1 and fake_encoder.image_calls == 1
    assert np.array_equal(m1.embeddings, m2.embeddings)
    t1 = store.text_embeddings(["a cup", "a bed", "a cup"])
    t2 = FrameStore(fake_encoder, cache_root=tmp_path, fetcher=FakeFetcher()).text_embeddings(["a bed"])
    assert fake_encoder.text_calls == 1 and np.allclose(t1[1], t2[0]) and np.allclose(t1[0], t1[2])


def test_semantic_topk_order_and_chronology():
    mem, q = make_memory([0.1, 0.9, 0.3, 0.8, 0.2])
    r = SemanticRetriever().retrieve(mem, q, 2)
    assert r.ranking == [1, 3] and r.frame_indices == [1, 3]
    assert r.scores[0] == pytest.approx(0.9, abs=1e-5)


def test_semantic_ties_deterministic():
    mem, q = make_memory([0.5, 0.5, 0.5, 0.5])
    assert SemanticRetriever().retrieve(mem, q, 2).ranking == [0, 1]


def test_semantic_skips_invalid_and_small_episode():
    mem, q = make_memory([0.9, 0.1, 0.8], valid=[False, True, True])
    r = SemanticRetriever().retrieve(mem, q, 8)
    assert r.frame_indices == [1, 2] and r.config["k_effective"] == 2


def test_k_must_be_positive():
    mem, q = make_memory([0.1, 0.2])
    with pytest.raises(ValueError):
        SemanticRetriever().retrieve(mem, q, 0)
