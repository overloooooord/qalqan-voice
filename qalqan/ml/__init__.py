"""Neural network architecture, hybrid deepfake detector, and training pipeline."""

from qalqan.ml.detector import DetectionVerdict, VoiceDeepfakeDetector

__all__ = [
    "DetectionVerdict",
    "VoiceDeepfakeDetector",
]

try:  # Optional PyTorch exports when the [ml] extra is installed
    from qalqan.ml.model import MFM1d, MFM2d, QalqanLCNN, load_checkpoint, save_checkpoint

    __all__ += ["MFM1d", "MFM2d", "QalqanLCNN", "load_checkpoint", "save_checkpoint"]
except ImportError:  # pragma: no cover - lightweight serverless runtime (e.g., Vercel)
    pass
