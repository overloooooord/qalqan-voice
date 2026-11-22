"""CLI utility to inspect a local audio file or run built-in synthetic/live demo checks.

Examples:
    python -m scripts.demo_cli --demo
    python -m scripts.demo_cli path/to/voice.ogg --lang kk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qalqan.audio.loader import load_audio_file
from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.ml.detector import VoiceDeepfakeDetector


def _print_verdict(title: str, verdict, as_json: bool = False) -> None:
    if as_json:
        payload = verdict.to_dict()
        payload.pop("spectrogram_preview", None)
        print(json.dumps({"target": title, **payload}, ensure_ascii=False, indent=2))
        return

    pct = verdict.probability * 100.0
    print(f"\n=== {title} ===")
    print(f"Verdict     : {verdict.label.upper()} ({pct:.1f}% synthetic)")
    print(f"Engine      : {verdict.engine_mode}")
    print(f"Duration    : {verdict.duration_sec:.2f}s | Latency: {verdict.latency_ms:.1f} ms")
    print("Reasons:")
    for r in verdict.reasons:
        print(f"  - {r}")
    m = verdict.metrics
    print(
        f"Metrics     : vocoder_6_8k={m['hf_vocoder_score']:.2f}, "
        f"breath={m['breath_pause_score']:.2f}, "
        f"resonance={m['resonance_smoothness_score']:.2f}, "
        f"flux={m['phase_stationarity_score']:.2f}, "
        f"snr={m['snr_estimate_db']:.1f} dB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QALQAN Voice command-line deepfake analyzer.")
    parser.add_argument("audio_path", nargs="?", type=Path, help="Path to WAV/OGG/MP3/WebM/M4A file.")
    parser.add_argument("--demo", action="store_true", help="Run built-in synthetic vs live demonstration.")
    parser.add_argument("--lang", choices=["ru", "kk", "en"], default="ru")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted text.")
    args = parser.parse_args()

    detector = VoiceDeepfakeDetector()

    if args.demo or args.audio_path is None:
        v_synth = detector.analyze_waveform(generate_synthetic_voice_sample(), sample_rate=16000, lang=args.lang)
        v_live = detector.analyze_waveform(generate_live_human_sample(), sample_rate=16000, lang=args.lang)
        _print_verdict("Demo: Synthetic Clone", v_synth, as_json=args.json)
        _print_verdict("Demo: Live Human Voice", v_live, as_json=args.json)
        return

    loaded = load_audio_file(args.audio_path)
    verdict = detector.analyze_loaded(loaded, lang=args.lang)
    _print_verdict(str(args.audio_path), verdict, as_json=args.json)


if __name__ == "__main__":
    main()
