"""Telegram bot for instant verification of forwarded voice messages (aiogram 3).

Usage:
    export QALQAN_TELEGRAM_TOKEN="123456:ABC..."
    python -m qalqan.bot.telegram_bot
"""

from __future__ import annotations

import asyncio
import io
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from qalqan.audio.loader import AudioLoadError
from qalqan.config import Settings
from qalqan.ml.detector import DetectionVerdict, VoiceDeepfakeDetector

logger = logging.getLogger(__name__)

_UI_TEXT: dict[str, dict[str, str]] = {
    "ru": {
        "welcome": (
            "🛡️ <b>QALQAN Voice</b> — детектор поддельных ИИ-голосов (RU / KZ).\n\n"
            "Перешлите мне подозрительное <b>голосовое сообщение</b>, <b>кружок</b> или <b>аудиофайл</b> — "
            "я проверю физику звука за 1–2 секунды и выдам вердикт.\n\n"
            "Команды:\n"
            "• /lang — переключить язык (RU / KZ / EN)\n"
            "• /help — как работает проверка"
        ),
        "help": (
            "🔍 <b>Как работает QALQAN Voice:</b>\n"
            "Мы анализируем не смысл слов, а физику звука:\n"
            "1. Микропаузы дыхания и динамику фраз\n"
            "2. Резонанс голосового тракта и микро-джиттер тона\n"
            "3. Следы нейровокодеров (ElevenLabs, RVC, VALL-E) в полосе 6–8 кГц\n\n"
            "Просто перешлите голосовое сообщение сюда."
        ),
        "checking": "⏳ Анализирую акустический профиль…",
        "synthetic_title": "🚨 ВНИМАНИЕ: СИНТЕТИЧЕСКИЙ ГОЛОС",
        "suspicious_title": "⚠️ ПОДОЗРИТЕЛЬНЫЙ ГОЛОС",
        "live_title": "✅ ЖИВОЙ ГОЛОС",
        "prob_label": "Вероятность синтеза",
        "duration_label": "Длительность",
        "latency_label": "Время проверки",
        "reasons_label": "Признаки",
        "err_decode": "❌ Не удалось обработать аудио: {err}",
    },
    "kk": {
        "welcome": (
            "🛡️ <b>QALQAN Voice</b> — жасанды ИИ-дауыстарды анықтау жүйесі (KZ / RU).\n\n"
            "Күмәнді <b>дауыстық хабарламаны</b> немесе <b>аудиофайлды</b> маған жіберіңіз — "
            "дыбыс физикасын 1–2 секундта тексеріп, нәтиже беремін.\n\n"
            "Командалар:\n"
            "• /lang — тілді ауыстыру (KZ / RU / EN)\n"
            "• /help — тексеру қалай жұмыс істейді"
        ),
        "help": (
            "🔍 <b>QALQAN Voice қалай жұмыс істейді:</b>\n"
            "Сөз мағынасын емес, дыбыстың физикалық сипаттамаларын тексереміз:\n"
            "1. Фразалар арасындағы табиғи тыныс алу\n"
            "2. Дауыс жолының резонансы мен тон өзгерісі\n"
            "3. 6–8 кГц диапазонындағы нейровокодер іздері"
        ),
        "checking": "⏳ Акустикалық профиль талдануда…",
        "synthetic_title": "🚨 НАЗАР АУДАРЫҢЫЗ: СИНТЕТИКАЛЫҚ ДАУЫС",
        "suspicious_title": "⚠️ КҮМӘНДІ ДАУЫС",
        "live_title": "✅ ТІРІ ДАУЫС",
        "prob_label": "Синтез ықтималдығы",
        "duration_label": "Ұзақтығы",
        "latency_label": "Тексеру уақыты",
        "reasons_label": "Белгілер",
        "err_decode": "❌ Аудионы өңдеу мүмкін болмады: {err}",
    },
    "en": {
        "welcome": (
            "🛡️ <b>QALQAN Voice</b> — Real-time AI voice deepfake detector (KZ / RU / EN).\n\n"
            "Forward a suspicious <b>voice message</b>, <b>video note</b>, or <b>audio file</b> "
            "to inspect its acoustic physics in 1–2 seconds.\n\n"
            "Commands:\n"
            "• /lang — switch language (RU / KZ / EN)\n"
            "• /help — how detection works"
        ),
        "help": (
            "🔍 <b>How QALQAN Voice works:</b>\n"
            "We analyze acoustic physics rather than transcript content:\n"
            "1. Inter-phrase micro-breaths & pause entropy\n"
            "2. Vocal-tract resonance & pitch jitter\n"
            "3. 6–8 kHz neural vocoder artifacts"
        ),
        "checking": "⏳ Analyzing acoustic signature…",
        "synthetic_title": "🚨 ALERT: SYNTHETIC VOICE",
        "suspicious_title": "⚠️ SUSPICIOUS VOICE",
        "live_title": "✅ LIVE HUMAN VOICE",
        "prob_label": "Synthetic probability",
        "duration_label": "Duration",
        "latency_label": "Latency",
        "reasons_label": "Forensic findings",
        "err_decode": "❌ Could not decode audio: {err}",
    },
}


def _risk_bar(prob: float, width: int = 10) -> str:
    filled = max(1, min(width, round(prob * width)))
    block = "🟥" if prob >= 0.65 else ("🟨" if prob >= 0.40 else "🟩")
    return block * filled + "⬜" * (width - filled)


def format_verdict_message(verdict: DetectionVerdict, lang: str = "ru") -> str:
    """Format a DetectionVerdict into a clean Telegram HTML card."""
    ui = _UI_TEXT.get(lang, _UI_TEXT["ru"])
    title_key = f"{verdict.label}_title"
    title = ui.get(title_key, ui["suspicious_title"])
    pct = verdict.probability * 100.0
    reasons_block = "\n".join(f"• {r}" for r in verdict.reasons)

    return (
        f"<b>{title}</b>\n\n"
        f"{_risk_bar(verdict.probability)}  <b>{pct:.1f}%</b>\n"
        f"{ui['prob_label']}: <code>{pct:.1f}%</code>\n"
        f"{ui['duration_label']}: <code>{verdict.duration_sec:.1f}s</code> · "
        f"{ui['latency_label']}: <code>{verdict.latency_ms:.0f} ms</code>\n\n"
        f"<b>{ui['reasons_label']}:</b>\n{reasons_block}"
    )


def build_router(detector: VoiceDeepfakeDetector, default_lang: str = "ru") -> Router:
    router = Router()
    user_langs: dict[int, str] = {}

    def _lang_for(message: Message) -> str:
        uid = message.from_user.id if message.from_user else 0
        return user_langs.get(uid, default_lang)

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        lang = _lang_for(message)
        await message.answer(_UI_TEXT[lang]["welcome"])

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        lang = _lang_for(message)
        await message.answer(_UI_TEXT[lang]["help"])

    @router.message(Command("lang"))
    async def cmd_lang(message: Message) -> None:
        uid = message.from_user.id if message.from_user else 0
        order = ["ru", "kk", "en"]
        cur = user_langs.get(uid, default_lang)
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "ru"
        user_langs[uid] = nxt
        labels = {"ru": "Русский 🇷🇺", "kk": "Қазақша 🇰🇿", "en": "English 🇬🇧"}
        await message.answer(f"🌐 Language switched to: <b>{labels[nxt]}</b>")

    @router.message(F.voice | F.audio | F.video_note | F.document)
    async def handle_audio(message: Message, bot: Bot) -> None:
        file_id: str | None = None
        filename = "voice.ogg"

        if message.voice:
            file_id = message.voice.file_id
            filename = "voice.ogg"
        elif message.audio:
            file_id = message.audio.file_id
            filename = message.audio.file_name or "audio.mp3"
        elif message.video_note:
            file_id = message.video_note.file_id
            filename = "video_note.mp4"
        elif message.document and (message.document.mime_type or "").startswith("audio/"):
            file_id = message.document.file_id
            filename = message.document.file_name or "audio.wav"

        if not file_id:
            return

        lang = _lang_for(message)
        ui = _UI_TEXT[lang]
        status_msg = await message.reply(ui["checking"])

        buf = io.BytesIO()
        await bot.download(file_id, destination=buf)
        raw_bytes = buf.getvalue()

        try:
            verdict = await asyncio.to_thread(detector.analyze_bytes, raw_bytes, filename, lang)
            text = format_verdict_message(verdict, lang=lang)
        except AudioLoadError as exc:
            text = ui["err_decode"].format(err=str(exc))

        await status_msg.edit_text(text)

    return router


async def run_bot(settings: Settings | None = None) -> None:
    cfg = settings or Settings.from_env()
    if not cfg.telegram_bot_token:
        raise RuntimeError(
            "QALQAN_TELEGRAM_TOKEN is not set. Export your BotFather token before launching the Telegram bot."
        )

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO))
    detector = VoiceDeepfakeDetector(settings=cfg)
    bot = Bot(
        token=cfg.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(build_router(detector, default_lang=cfg.default_lang))
    logger.info("Starting QALQAN Voice Telegram bot (engine=%s)", detector.engine_mode)
    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
