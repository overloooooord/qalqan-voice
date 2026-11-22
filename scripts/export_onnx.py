"""Export a QalqanLCNN PyTorch checkpoint to ONNX for low-latency on-premise / C++ inference.

Usage:
    python -m scripts.export_onnx --checkpoint models/qalqan_lcnn.pt --out models/qalqan_lcnn.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qalqan.ml.model import QalqanLCNN, load_checkpoint


def export_to_onnx(checkpoint: Path | None, out_path: Path, opset: int = 17) -> Path:
    if checkpoint and checkpoint.is_file():
        model, _meta = load_checkpoint(checkpoint, device="cpu")
    else:
        model = QalqanLCNN()
        model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_lfcc = torch.randn(1, 1, 60, 300, dtype=torch.float32)
    dummy_phys = torch.zeros(1, 4, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_lfcc, dummy_phys),
        str(out_path),
        input_names=["lfcc", "physics"],
        output_names=["logit"],
        dynamic_axes={
            "lfcc": {0: "batch", 3: "time_frames"},
            "physics": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=opset,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QalqanLCNN to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint (optional for untrained graph export).")
    parser.add_argument("--out", type=Path, default=Path("models/qalqan_lcnn.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = export_to_onnx(args.checkpoint, args.out, opset=args.opset)
    print(f"Exported ONNX model to {out}")


if __name__ == "__main__":
    main()
