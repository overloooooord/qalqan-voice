(() => {
  const I18N = {
    ru: {
      badge: "DEFENSE TECH · КИБЕРБЕЗОПАСНОСТЬ",
      hero_desc:
        "Детектор поддельных голосов на казахском и русском. Проверяет звонок и голосовое сообщение в реальном времени по физике звука.",
      hero_quote:
        "Слушаем не слова, а физику звука: микропаузы дыхания, резонанс голосового тракта и следы нейровокодеров в полосе 6–8 кГц.",
      st_dash_title: "Веб-дашборд и REST / WebSocket API",
      st_ml_title: "Акустическая физика + PyTorch LCNN (RU / KZ)",
      st_ml_sub: "LFCC 60-dim + анализ 6–8 кГц вокодера и джиттера тона",
      drop_title: "Проверить аудио или голосовое (.ogg / .wav / .mp3)",
      drop_sub: "Перетащите файл сюда или нажмите для выбора (до 25 МБ)",
      spec_idle: "Нажмите «Симуляция», включите микрофон (M) или загрузите голосовое",
      prob_short: "вероятность синтеза",
      verdict_ready: "ОЖИДАНИЕ АУДИО",
      verdict_synthetic: "СИНТЕТИЧЕСКИЙ ГОЛОС",
      verdict_suspicious: "ПОДОЗРИТЕЛЬНЫЙ ГОЛОС",
      verdict_live: "ЖИВОЙ ГОЛОС",
      reason_idle: "Система готова к проверке потока или файла в реальном времени",
      m_vocoder: "Вокодер 6–8 кГц",
      m_breath: "Микропаузы дыхания",
      m_resonance: "Ровность резонанса",
      m_flux: "Спектр. поток",
      history_title: "История проверок",
      history_clear: "Очистить",
      history_empty: "Пока нет проверок",
      btn_mic: "Микрофон",
      btn_mic_stop: "Остановить",
      btn_sim_synth: "Симуляция: ИИ",
      btn_sim_live: "Симуляция: Живой",
      hist_synthetic: "синтез",
      hist_suspicious: "подозр.",
      hist_live: "живой",
      mode_ready: "режим: готов",
      mode_demo: "режим: демо-симуляция",
      mode_mic: "режим: микрофон (WebSocket)",
      mode_file: "режим: анализ файла",
      analyzing: "АНАЛИЗ АУДИО…",
    },
    kk: {
      badge: "DEFENSE TECH · КИБЕРҚАУІПСІЗДІК",
      hero_desc:
        "Қазақ және орыс тілдеріндегі жасанды дауыстарды анықтайтын детектор. Қоңырау мен дауыстық хабарламаны нақты уақытта тексереді.",
      hero_quote:
        "Сөз мағынасын емес, дыбыс физикасын тыңдаймыз: тыныс алу микропаузалары, дауыс жолының резонансы және 6–8 кГц вокодер іздері.",
      st_dash_title: "Веб-дашборд және REST / WebSocket API",
      st_ml_title: "Акустикалық физика + PyTorch LCNN (KZ / RU)",
      st_ml_sub: "LFCC 60-dim + 6–8 кГц вокодер және тон джиттерін талдау",
      drop_title: "Аудио немесе дауыстық хабарламаны тексеру (.ogg / .wav / .mp3)",
      drop_sub: "Файлды осында сүйреңіз немесе таңдау үшін басыңыз (25 МБ дейін)",
      spec_idle: "«Симуляция» батырмасын басыңыз, микрофонды қосыңыз (M) немесе файл жүктеңіз",
      prob_short: "синтез ықтималдығы",
      verdict_ready: "АУДИО КҮТІЛУДЕ",
      verdict_synthetic: "СИНТЕТИКАЛЫҚ ДАУЫС",
      verdict_suspicious: "КҮМӘНДІ ДАУЫС",
      verdict_live: "ТІРІ ДАУЫС",
      reason_idle: "Жүйе нақты уақыттағы тексеруге дайын",
      m_vocoder: "Вокодер 6–8 кГц",
      m_breath: "Тыныс микропаузасы",
      m_resonance: "Резонанс тегістігі",
      m_flux: "Спектрлік ағын",
      history_title: "Тексеру тарихы",
      history_clear: "Тазарту",
      history_empty: "Әзірге тексеру жоқ",
      btn_mic: "Микрофон",
      btn_mic_stop: "Тоқтату",
      btn_sim_synth: "Симуляция: ИИ",
      btn_sim_live: "Симуляция: Тірі",
      hist_synthetic: "синтез",
      hist_suspicious: "күмәнді",
      hist_live: "тірі",
      mode_ready: "режим: дайын",
      mode_demo: "режим: демо-симуляция",
      mode_mic: "режим: микрофон (WebSocket)",
      mode_file: "режим: файлды талдау",
      analyzing: "ТАЛДАУ ЖҮРІП ЖАТЫР…",
    },
    en: {
      badge: "DEFENSE TECH · CYBERSECURITY",
      hero_desc:
        "Real-time voice deepfake detector for Kazakh, Russian, and code-switched speech. Verifies calls and voice notes via acoustic physics.",
      hero_quote:
        "We analyze the physics of sound, not transcripts: inter-phrase micro-breaths, vocal-tract resonance, and 6–8 kHz neural vocoder artifacts.",
      st_dash_title: "Web Dashboard & REST / WebSocket API",
      st_ml_title: "Acoustic Physics + PyTorch LCNN (KZ / RU)",
      st_ml_sub: "60-dim LFCC + 6–8 kHz vocoder & pitch micro-jitter forensics",
      drop_title: "Verify audio or voice note (.ogg / .wav / .mp3)",
      drop_sub: "Drag & drop an audio file here or click to browse (up to 25 MB)",
      spec_idle: "Click Simulate, toggle Microphone (M), or drop an audio file",
      prob_short: "synthetic score",
      verdict_ready: "AWAITING AUDIO",
      verdict_synthetic: "SYNTHETIC VOICE",
      verdict_suspicious: "SUSPICIOUS VOICE",
      verdict_live: "LIVE HUMAN VOICE",
      reason_idle: "Ready for real-time audio stream or voice note inspection",
      m_vocoder: "6–8 kHz Vocoder",
      m_breath: "Micro-Breath Gap",
      m_resonance: "Resonance Smoothness",
      m_flux: "Spectral Flux",
      history_title: "Verification Log",
      history_clear: "Clear",
      history_empty: "No checks yet",
      btn_mic: "Microphone",
      btn_mic_stop: "Stop Mic",
      btn_sim_synth: "Simulate: AI",
      btn_sim_live: "Simulate: Live",
      hist_synthetic: "synthetic",
      hist_suspicious: "suspicious",
      hist_live: "live",
      mode_ready: "mode: ready",
      mode_demo: "mode: demo simulation",
      mode_mic: "mode: live microphone (WS)",
      mode_file: "mode: file upload",
      analyzing: "ANALYZING AUDIO…",
    },
  };

  let currentLang = "ru";
  let isRecording = false;
  let audioCtx = null;
  let mediaStream = null;
  let scriptNode = null;
  let analyser = null;
  let ws = null;
  let animFrameId = null;
  let lastSpecMatrix = null;

  const $ = (id) => document.getElementById(id);
  const RING_CIRC = 2 * Math.PI * 50;

  function t(key) {
    return (I18N[currentLang] && I18N[currentLang][key]) || I18N.ru[key] || key;
  }

  function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (key) el.textContent = t(key);
    });
    $("micBtnText").textContent = isRecording ? t("btn_mic_stop") : t("btn_mic");
  }

  // Adaptive spectrogram palette (light mode: Slide 6 lavender->indigo->crimson; dark mode: midnight->indigo->magenta->coral)
  function heatColor(v) {
    const x = Math.max(0, Math.min(1, v));
    if (document.body.classList.contains("dark")) {
      if (x < 0.35) {
        const p = x / 0.35;
        return [Math.round(14 + p * 60), Math.round(20 + p * 44), Math.round(46 + p * 120)];
      }
      if (x < 0.7) {
        const p = (x - 0.35) / 0.35;
        return [Math.round(74 + p * 138), Math.round(64 + p * 20), Math.round(166 + p * 36)];
      }
      const p = (x - 0.7) / 0.3;
      return [Math.round(212 + p * 39), Math.round(84 + p * 29), Math.round(202 - p * 70)];
    }
    if (x < 0.35) {
      const p = x / 0.35;
      const r = Math.round(234 - p * 96);
      const g = Math.round(236 - p * 86);
      const b = Math.round(248 - p * 12);
      return [r, g, b];
    }
    if (x < 0.7) {
      const p = (x - 0.35) / 0.35;
      const r = Math.round(138 + p * 74);
      const g = Math.round(150 - p * 84);
      const b = Math.round(236 - p * 44);
      return [r, g, b];
    }
    const p = (x - 0.7) / 0.3;
    const r = Math.round(212 + p * 27);
    const g = Math.round(66 - p * 34);
    const b = Math.round(192 - p * 110);
    return [r, g, b];
  }

  function renderSpectrogramMatrix(matrix) {
    const canvas = $("specCanvas");
    const ctx = canvas.getContext("2d");
    if (!matrix || !matrix.length || !matrix[0].length) return;
    lastSpecMatrix = matrix;

    $("specOverlay").classList.add("hidden");
    const rows = matrix.length;
    const cols = matrix[0].length;
    const cw = canvas.width;
    const ch = canvas.height;
    const cellW = cw / cols;
    const cellH = ch / rows;

    ctx.clearRect(0, 0, cw, ch);
    for (let r = 0; r < rows; r++) {
      // High frequency at top, low frequency at bottom
      const y = (rows - 1 - r) * cellH;
      for (let c = 0; c < cols; c++) {
        const [red, grn, blu] = heatColor(matrix[r][c]);
        ctx.fillStyle = `rgb(${red},${grn},${blu})`;
        ctx.fillRect(c * cellW, y, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
      }
    }
  }

  function setBar(barId, valId, score) {
    const pct = Math.round((score || 0) * 100);
    $(barId).style.width = `${pct}%`;
    $(valId).textContent = `${pct}%`;
  }

  function renderVerdict(data) {
    const card = $("monitorCard");
    card.classList.remove("state-synthetic", "state-suspicious", "state-live");
    card.classList.add(`state-${data.label}`);

    const pct = Math.round(data.probability * 100);
    $("ringPct").textContent = `${pct}%`;
    const offset = RING_CIRC * (1 - data.probability);
    $("ringProgress").style.strokeDashoffset = `${offset}`;

    const titleMap = {
      synthetic: t("verdict_synthetic"),
      suspicious: t("verdict_suspicious"),
      live: t("verdict_live"),
    };
    $("verdictTitle").textContent = titleMap[data.label] || data.label.toUpperCase();
    $("latencyChip").textContent = `${Math.round(data.latency_ms)} ms`;
    $("sourceChip").textContent = `${data.source} · ${data.duration_sec}s`;

    const ul = $("reasonList");
    ul.innerHTML = "";
    (data.reasons || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      ul.appendChild(li);
    });

    const banner = $("quoteBanner");
    if (data.transcript_hint) {
      banner.textContent = data.transcript_hint;
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
    }

    if (data.metrics) {
      setBar("barVocoder", "valVocoder", data.metrics.hf_vocoder_score);
      setBar("barBreath", "valBreath", data.metrics.breath_pause_score);
      setBar("barResonance", "valResonance", data.metrics.resonance_smoothness_score);
      setBar("barFlux", "valFlux", data.metrics.phase_stationarity_score);
    }

    if (data.spectrogram_preview) {
      renderSpectrogramMatrix(data.spectrogram_preview);
    }
  }

  async function refreshHistory() {
    try {
      const res = await fetch("/api/v1/history?limit=10");
      if (!res.ok) return;
      const items = await res.json();
      const box = $("historyList");
      if (!items.length) {
        box.innerHTML = `<div class="history-empty">${t("history_empty")}</div>`;
        return;
      }
      box.innerHTML = "";
      items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "history-row";
        const labelWord = t(`hist_${item.label}`);
        const pct = Math.round(item.probability * 100);
        row.innerHTML = `
          <div class="history-left">
            <span class="mono">${item.timestamp}</span>
            <span>·</span>
            <span class="hist-badge-${item.label}">${labelWord}</span>
            <span style="color:var(--muted);font-size:11.5px">${item.source}</span>
          </div>
          <span class="mono hist-badge-${item.label}">${pct}%</span>
        `;
        box.appendChild(row);
      });
    } catch (_) {
      // Ignore transient polling errors
    }
  }

  async function loadHealth() {
    try {
      const res = await fetch("/api/v1/health");
      if (!res.ok) throw new Error("offline");
      const h = await res.json();
      $("healthTag").textContent = "ONLINE";
      $("engineTag").textContent = h.model_loaded ? "HYBRID LCNN" : "PHYSICS DSP";
      $("healthMeta").textContent =
        `FastAPI v${h.version} · ${h.sample_rate / 1000} kHz · engine: ${h.engine_mode} · FFmpeg: ${h.ffmpeg_available ? "OK" : "WAV only"}`;
    } catch (_) {
      $("healthTag").textContent = "OFFLINE";
      $("healthMeta").textContent = "API недоступен";
    }
  }

  async function runSimulation(scenario) {
    const btns = [$("simSynthBtn"), $("simLiveBtn")];
    btns.forEach((b) => (b.disabled = true));
    $("verdictTitle").textContent = t("analyzing");
    $("modeFoot").textContent = t("mode_demo");

    try {
      const res = await fetch("/api/v1/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, lang: currentLang }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderVerdict(data);
      await refreshHistory();
    } catch (err) {
      $("verdictTitle").textContent = `Ошибка: ${err.message}`;
    } finally {
      btns.forEach((b) => (b.disabled = false));
    }
  }

  async function uploadAudioFile(file) {
    if (!file) return;
    $("verdictTitle").textContent = t("analyzing");
    $("modeFoot").textContent = t("mode_file");

    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`/api/v1/analyze?lang=${encodeURIComponent(currentLang)}`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      renderVerdict(data);
      await refreshHistory();
    } catch (err) {
      $("verdictTitle").textContent = `Ошибка: ${err.message}`;
    }
  }

  // Real-time Microphone -> WebSocket PCM16 streaming + live scrolling spectrogram
  function floatTo16BitPCM(float32Arr) {
    const buf = new ArrayBuffer(float32Arr.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32Arr.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Arr[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buf;
  }

  function startLiveCanvasScroll() {
    const canvas = $("specCanvas");
    const ctx = canvas.getContext("2d");
    const bins = new Uint8Array(analyser.frequencyBinCount);
    $("specOverlay").classList.add("hidden");

    const draw = () => {
      if (!isRecording || !analyser) return;
      analyser.getByteFrequencyData(bins);
      const cw = canvas.width;
      const ch = canvas.height;
      // Shift existing spectrogram left by 3px
      ctx.drawImage(canvas, 3, 0, cw - 3, ch, 0, 0, cw - 3, ch);
      const step = ch / 48;
      for (let i = 0; i < 48; i++) {
        const binIdx = Math.min(bins.length - 1, Math.floor((i / 48) * (bins.length * 0.7)));
        const norm = bins[binIdx] / 255;
        const [r, g, b] = heatColor(norm);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(cw - 3, ch - (i + 1) * step, 3, Math.ceil(step) + 0.5);
      }
      animFrameId = requestAnimationFrame(draw);
    };
    animFrameId = requestAnimationFrame(draw);
  }

  async function toggleMic() {
    if (isRecording) {
      stopMic();
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioCtx.createMediaStreamSource(mediaStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);

      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(
        `${proto}//${location.host}/ws/stream?lang=${encodeURIComponent(currentLang)}&sample_rate=${audioCtx.sampleRate}`
      );
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.type === "verdict") {
          renderVerdict(msg);
          refreshHistory();
        } else if (msg.type === "buffering") {
          $("modeFoot").textContent = `${t("mode_mic")} · ${msg.buffered_sec}s`;
        }
      };

      scriptNode.onaudioprocess = (ev) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = ev.inputBuffer.getChannelData(0);
        ws.send(floatTo16BitPCM(input));
      };

      src.connect(analyser);
      analyser.connect(scriptNode);
      scriptNode.connect(audioCtx.destination);

      isRecording = true;
      $("micBtn").classList.add("recording");
      $("micBtnText").textContent = t("btn_mic_stop");
      $("modeFoot").textContent = t("mode_mic");
      $("sourceChip").textContent = "M · микрофон (LIVE)";
      startLiveCanvasScroll();
    } catch (err) {
      $("verdictTitle").textContent = `Микрофон недоступен: ${err.message}`;
      stopMic();
    }
  }

  function stopMic() {
    isRecording = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
    if (analyser) { analyser.disconnect(); analyser = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach((tr) => tr.stop()); mediaStream = null; }
    if (ws) { ws.close(); ws = null; }
    $("micBtn").classList.remove("recording");
    $("micBtnText").textContent = t("btn_mic");
    $("modeFoot").textContent = t("mode_ready");
  }

  // Bind events
  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".lang-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentLang = btn.dataset.lang;
      applyI18n();
      refreshHistory();
    });
  });

  $("themeToggle").addEventListener("click", () => {
    document.body.classList.toggle("dark");
    if (lastSpecMatrix && !isRecording) {
      renderSpectrogramMatrix(lastSpecMatrix);
    }
  });

  $("simSynthBtn").addEventListener("click", () => runSimulation("synthetic"));
  $("simLiveBtn").addEventListener("click", () => runSimulation("live"));
  $("micBtn").addEventListener("click", () => toggleMic());

  $("clearHistoryBtn").addEventListener("click", async () => {
    await fetch("/api/v1/history", { method: "DELETE" });
    refreshHistory();
  });

  const fileInput = $("audioFileInput");
  const dropzone = $("dropzone");
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      uploadAudioFile(fileInput.files[0]);
      fileInput.value = "";
    }
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadAudioFile(file);
  });

  // Keyboard shortcut 'M' to toggle microphone (matches Slide 6 "M · микрофон")
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key.toLowerCase() === "m") toggleMic();
  });

  applyI18n();
  loadHealth();
  refreshHistory();
})();
