"""Unit tests for the Telegram bot verdict card formatter."""

from __future__ import annotations

from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.bot.telegram_bot import format_verdict_message
from qalqan.config import Settings
from qalqan.ml.detector import VoiceDeepfakeDetector


def test_telegram_verdict_formatting_multilingual():
    detector = VoiceDeepfakeDetector(Settings(model_path=None))

    v_synth = detector.analyze_waveform(generate_synthetic_voice_sample(), lang="ru")
    msg_ru = format_verdict_message(v_synth, lang="ru")
    assert "СИНТЕТИЧЕСКИЙ ГОЛОС" in msg_ru
    assert "🟥" in msg_ru

    v_live_kk = detector.analyze_waveform(generate_live_human_sample(), lang="kk")
    msg_kk = format_verdict_message(v_live_kk, lang="kk")
    assert "ТІРІ ДАУЫС" in msg_kk
    assert "🟩" in msg_kk

    v_synth_en = detector.analyze_waveform(generate_synthetic_voice_sample(), lang="en")
    msg_en = format_verdict_message(v_synth_en, lang="en")
    assert "SYNTHETIC VOICE" in msg_en
