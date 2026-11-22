"""Unit tests for QalqanLCNN PyTorch architecture, checkpointing, and VoiceDeepfakeDetector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.config import Settings
from qalqan.ml.detector import VoiceDeepfakeDetector
from qalqan.ml.model import QalqanLCNN, load_checkpoint, save_checkpoint
from qalqan.ml.train import _build_bootstrap_dataset, compute_eer, train_model


def test_qalqan_lcnn_forward_variable_length():
    model = QalqanLCNN()
    model.eval()
    for t_frames in (64, 150, 320):
        lfcc = torch.randn(2, 1, 60, t_frames)
        phys = torch.rand(2, 4)
        out = model(lfcc, phys)
        assert out.shape == (2, 1)
        assert torch.all(torch.isfinite(out))


def test_checkpoint_save_and_load(tmp_path: Path):
    model = QalqanLCNN()
    ckpt = tmp_path / "qalqan_test.pt"
    save_checkpoint(model, ckpt, metrics={"val_eer": 0.05})

    loaded, meta = load_checkpoint(ckpt)
    assert meta["arch"] == "QalqanLCNN"
    assert meta["metrics"]["val_eer"] == 0.05

    x = torch.randn(1, 1, 60, 120)
    p = torch.zeros(1, 4)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(x, p), loaded(x, p), atol=1e-6)


def test_detector_physics_mode_classifications_and_languages():
    detector = VoiceDeepfakeDetector(Settings(model_path=None))
    assert detector.engine_mode == "acoustic-physics"

    for lang in ("ru", "kk", "en"):
        v_synth = detector.analyze_waveform(generate_synthetic_voice_sample(), lang=lang)
        v_live = detector.analyze_waveform(generate_live_human_sample(), lang=lang)

        assert v_synth.label == "synthetic"
        assert v_synth.probability >= 0.70
        assert len(v_synth.reasons) >= 1

        assert v_live.label == "live"
        assert v_live.probability <= 0.30
        assert len(v_live.reasons) >= 1


def test_bootstrap_training_and_hybrid_inference(tmp_path: Path):
    items = _build_bootstrap_dataset(n_per_class=6, sr=16000)
    ckpt = tmp_path / "hybrid.pt"
    metrics = train_model(items=items, out_path=ckpt, epochs=2, batch_size=4)
    assert ckpt.is_file()
    assert "val_eer" in metrics

    detector = VoiceDeepfakeDetector(Settings(model_path=ckpt))
    assert detector.engine_mode == "hybrid-lcnn"

    verdict = detector.analyze_waveform(generate_synthetic_voice_sample())
    assert verdict.metrics["neural_probability"] is not None
    assert 0.0 <= verdict.probability <= 1.0


def test_compute_eer():
    labels = np.array([0, 0, 0, 1, 1, 1])
    perfect_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_eer(labels, perfect_scores) == 0.0
