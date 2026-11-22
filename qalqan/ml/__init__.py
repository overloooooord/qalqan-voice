"""Neural network architecture, hybrid deepfake detector, and training pipeline."""

from qalqan.ml.detector import DetectionVerdict, VoiceDeepfakeDetector
from qalqan.ml.model import MFM1d, MFM2d, QalqanLCNN, load_checkpoint, save_checkpoint

__all__ = [
    "DetectionVerdict",
    "MFM1d",
    "MFM2d",
    "QalqanLCNN",
    "VoiceDeepfakeDetector",
    "load_checkpoint",
    "save_checkpoint",
]
