"""Training and evaluation pipeline for QalqanLCNN (RU/KZ voice deepfake detection).

Supports:
1. Real audio folder datasets:
     data_dir/
       real/*.wav|ogg|mp3|flac
       fake/*.wav|ogg|mp3|flac
2. Bootstrap self-verification mode (`--bootstrap-demo`) for CI / pipeline smoke tests.
3. Telephony & messenger channel augmentations (narrowband bandpass, street noise, gain jitter).

Usage:
    python -m qalqan.ml.train --data-dir ./dataset --epochs 20 --out models/qalqan_lcnn.pt
    python -m qalqan.ml.train --bootstrap-demo --epochs 5 --out models/qalqan_lcnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from qalqan.audio.features import extract_acoustic_features
from qalqan.audio.loader import load_audio_file
from qalqan.audio.synth_samples import bandpass_filter, generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.ml.model import QalqanLCNN, save_checkpoint

SUPPORTED_EXTS = {".wav", ".ogg", ".oga", ".mp3", ".flac", ".m4a", ".webm"}


def augment_channel(samples: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate real-world telephony, WhatsApp compression, and street/room noise."""
    x = samples.astype(np.float32).copy()

    # Random gain jitter
    x *= float(rng.uniform(0.65, 1.15))

    # 35% chance: narrowband telephony/WhatsApp bandpass (300 - 3400 Hz)
    if rng.random() < 0.35:
        x = bandpass_filter(x, sr=sr, low=300.0, high=3400.0)

    # 50% chance: additive background noise at 12-30 dB SNR
    if rng.random() < 0.50:
        sig_power = float(np.mean(x * x)) + 1e-8
        snr_db = float(rng.uniform(12.0, 30.0))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
        x = x + noise

    return np.clip(x, -1.0, 1.0)


def _pad_or_crop_lfcc(lfcc: np.ndarray, target_frames: int = 300) -> np.ndarray:
    n_dims, n_frames = lfcc.shape
    if n_frames == target_frames:
        return lfcc
    if n_frames > target_frames:
        return lfcc[:, :target_frames]
    pad_w = target_frames - n_frames
    return np.pad(lfcc, ((0, 0), (0, pad_w)), mode="constant")


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) in [0, 1] without external dependencies."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.all(labels == 0) or np.all(labels == 1):
        return 0.0
    thresholds = np.unique(scores)
    best_diff = 1.0
    eer = 0.5
    pos = labels == 1
    neg = labels == 0
    for thr in thresholds:
        far = float(np.mean(scores[neg] >= thr))
        frr = float(np.mean(scores[pos] < thr))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = 0.5 * (far + frr)
    return float(eer)


class VoiceSpoofDataset(Dataset):
    """PyTorch dataset of (lfcc, physics_vector, label) tensors."""

    def __init__(
        self,
        items: list[tuple[np.ndarray, int]],
        sr: int = 16000,
        target_frames: int = 300,
        augment: bool = False,
        seed: int = 42,
    ):
        self.items = items
        self.sr = sr
        self.target_frames = target_frames
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform, label = self.items[idx]
        if self.augment:
            waveform = augment_channel(waveform, sr=self.sr, rng=self.rng)
        feat = extract_acoustic_features(waveform, sr=self.sr)
        lfcc = _pad_or_crop_lfcc(feat.lfcc, target_frames=self.target_frames)
        # Per-utterance CMVN normalization
        lfcc = (lfcc - lfcc.mean(axis=1, keepdims=True)) / (lfcc.std(axis=1, keepdims=True) + 1e-6)
        return (
            torch.from_numpy(lfcc).unsqueeze(0),
            torch.from_numpy(feat.physics_vector()),
            torch.tensor([float(label)], dtype=torch.float32),
        )


def _load_folder_dataset(data_dir: Path, sr: int = 16000) -> list[tuple[np.ndarray, int]]:
    items: list[tuple[np.ndarray, int]] = []
    for folder_name, label in [("real", 0), ("fake", 1)]:
        sub = data_dir / folder_name
        if not sub.is_dir():
            raise FileNotFoundError(f"Expected directory '{sub}' containing {folder_name} audio files.")
        for path in sorted(sub.rglob("*")):
            if path.suffix.lower() in SUPPORTED_EXTS:
                loaded = load_audio_file(path, target_sr=sr, max_seconds=15.0)
                items.append((loaded.samples, label))
    if not items:
        raise ValueError(f"No supported audio files found under {data_dir}.")
    return items


def _build_bootstrap_dataset(n_per_class: int = 24, sr: int = 16000) -> list[tuple[np.ndarray, int]]:
    items: list[tuple[np.ndarray, int]] = []
    for i in range(n_per_class):
        items.append((generate_live_human_sample(duration_sec=3.0, sr=sr, seed=100 + i), 0))
        items.append((generate_synthetic_voice_sample(duration_sec=3.0, sr=sr, seed=500 + i), 1))
    return items


def train_model(
    items: list[tuple[np.ndarray, int]],
    out_path: Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    sr: int = 16000,
    seed: int = 42,
) -> dict[str, float]:
    """Train QalqanLCNN and save the best validation checkpoint."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(items))
    split = max(2, int(len(items) * 0.8))
    train_items = [items[i] for i in indices[:split]]
    val_items = [items[i] for i in indices[split:]] or train_items[:4]

    train_ds = VoiceSpoofDataset(train_items, sr=sr, augment=True, seed=seed)
    val_ds = VoiceSpoofDataset(val_items, sr=sr, augment=False, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = QalqanLCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_eer = 1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for lfcc, phys, y in train_loader:
            lfcc, phys, y = lfcc.to(device), phys.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(lfcc, phys)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * lfcc.size(0)
        train_loss /= max(1, len(train_ds))

        model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []
        with torch.inference_mode():
            for lfcc, phys, y in val_loader:
                lfcc, phys = lfcc.to(device), phys.to(device)
                probs = torch.sigmoid(model(lfcc, phys)).squeeze(-1).cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])
                all_scores.extend(probs.tolist())
                all_labels.extend(y.squeeze(-1).int().tolist())

        scores_arr = np.asarray(all_scores)
        labels_arr = np.asarray(all_labels)
        val_acc = float(np.mean((scores_arr >= 0.5) == (labels_arr == 1)))
        val_eer = compute_eer(labels_arr, scores_arr)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f} | val_eer={val_eer:.4f}")

        if val_eer <= best_eer:
            best_eer = val_eer
            best_metrics = {
                "val_eer": round(val_eer, 4),
                "val_accuracy": round(val_acc, 4),
                "train_loss": round(train_loss, 4),
                "epochs": epoch,
            }
            save_checkpoint(model, out_path, metrics=best_metrics, sample_rate=sr)

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QALQAN Voice LCNN deepfake detector.")
    parser.add_argument("--data-dir", type=Path, help="Folder containing real/ and fake/ audio subdirectories.")
    parser.add_argument("--bootstrap-demo", action="store_true", help="Train on synthetic/organic acoustic fixtures to verify pipeline.")
    parser.add_argument("--out", type=Path, default=Path("models/qalqan_lcnn.pt"), help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.data_dir:
        items = _load_folder_dataset(args.data_dir)
    elif args.bootstrap_demo:
        items = _build_bootstrap_dataset(n_per_class=20)
    else:
        parser.error("Specify either --data-dir PATH or --bootstrap-demo.")

    metrics = train_model(items=items, out_path=args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"Saved checkpoint to {args.out} | metrics={metrics}")


if __name__ == "__main__":
    main()
