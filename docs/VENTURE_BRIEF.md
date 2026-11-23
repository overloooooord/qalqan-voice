# QALQAN Voice — Venture & Market Evaluation Brief

> Ready-to-submit startup profile designed for venture evaluators, accelerators, grants, and B2B pilot discussions.

## One-Liner
**QALQAN Voice** is a sovereign, on-premise AI voice anti-spoofing and deepfake firewall built for **Kazakh, Russian, Turkic, and bilingual code-switched speech** — screening live calls and messenger voice notes in **< 35 ms on CPU** for banks, telecom operators, enterprise contact centers, and government agencies across **Central Asia, CIS, and Turkic/MENA markets**.

---

## 1. Problem & Urgency
- **Voice cloning has hit zero marginal cost:** Open-source and commercial models (ElevenLabs, XTTS-v2, RVC, VALL-E) clone any human voice from **3 seconds of audio for $0**, fueling CEO fraud ("fake director" wire transfers), "relative in distress" phone scams, and spoofed government/executive voice commands.
- **Enterprise & national fraud scale:** Feedzai's 2025 AI Fraud Report shows **over 50% of banking scams already involve AI**; Arup lost **$25M** in a single deepfake executive call; in Kazakhstan alone, the **National Bank of Kazakhstan (NBK) Anti-Fraud Center** (launched July 2024, connecting **250+ financial, telecom, and law-enforcement organizations**) has already blocked **28.3 billion KZT (~$56M)** in suspicious operations, while Tele2/Altel and local regulators repeatedly warn of surging AI-impersonation attacks.
- **Why existing US/EU detectors fail locally (The Structural Gap):**
  1. **Data Residency & Air-Gap Blockers:** US cloud SaaS vendors (Pindrop, Resemble AI) require routing sensitive voice audio to foreign servers — violating **Kazakhstan's Law on Personal Data Localization**, banking secrecy rules, and defense/gov air-gap policies across Central Asia & CIS.
  2. **Bilingual & Code-Switching Blind Spot:** English-centric detectors experience elevated false-positive rates on **agglutinative Turkic phonetics (Kazakh, Uzbek, Azerbaijani, Turkish)** and **intra-sentence Russian–Kazakh code-switching** (*«Сәлем, срочно переведи, мен кейін бәрін түсіндіремін»*), plus aggressive WhatsApp/Telegram Opus compression.
  3. **Messenger Channel Gap:** Legacy call-center tools (Pindrop, Nuance) ignore **WhatsApp and Telegram voice notes**, where the majority of regional CEO-fraud and citizen scams originate.

---

## 2. Product & Technology (Working MVP Live Today)
Public repository & verified MVP: **https://github.com/overloooooord/qalqan-voice** (CI passing, 16/16 automated tests, Docker on-premise image, Vercel-ready serverless API).

**Core Innovation — Physics of Sound + Compact Neural LCNN:**
Instead of relying on language-dependent speech-to-text transcripts, QALQAN Voice analyzes **language-agnostic acoustic physics** fused with a compact neural classifier:
1. **60-dim LFCC (Linear Frequency Cepstral Coefficients)** preserving 4–8 kHz linear resolution where neural vocoders leave phase and upsampling artifacts.
2. **4 Acoustic-Physics Forensics:**
   - `hf_vocoder_score` (6–8 kHz neural vocoder banding & temporal stationarity)
   - `breath_pause_score` (inter-phrase inhalation micro-breaths & pause-duration entropy vs. gated digital silence)
   - `resonance_smoothness_score` (vocal-tract formant drift & $F_0$ pitch micro-jitter)
   - `phase_stationarity_score` (harmonic/spectral flux regularity)
   - **Automatic Narrowband Codec Adaptation** for 8 kHz SIP telephony and 16 kbps WhatsApp Opus.
3. **PyTorch `QalqanLCNN` (Light CNN + Max-Feature-Map activations) + ONNX Export:** Compact CPU-native architecture trained with telephony/WhatsApp/noise augmentations (`qalqan.ml.train`), achieving **19–31 ms p50 latency per 4-second audio clip on a single CPU core (130×–208× faster than real-time)** across Clean 16 kHz, Telegram `.ogg` Opus 24k, WhatsApp `.ogg` Opus 16k, 15 dB noise, and 8 kHz SIP narrowband (`docs/BENCHMARK.md`).

**Three Integrated Deployment Form Factors:**
1. **B2B/B2G REST, WebSocket & SIP Sidecar API** for banks, contact centers, and telecom carriers (inline call risk scoring + fraud-team webhook alerts).
2. **Air-Gapped On-Premise Docker / ONNX Container** (~78 MB core runtime, CPU-only, zero outbound internet, zero raw audio disk persistence — RAM-only processing).
3. **Free Citizen & Executive Telegram / WhatsApp Bot (`aiogram 3`) + Live Web Dashboard (RU / KZ / EN)** for instant voice-note screening.

---

## 3. Market Size (Venture-Scale Bottom-Up & Top-Down)
- **Global TAM — $4.8B+ (30–38% CAGR):** Global voice anti-spoofing, deepfake audio detection, and voice-fraud prevention across banking, telecom, contact centers, and defense security (validated by Pindrop's $100M Series D / 5.3B calls analyzed across 8 of top 10 US banks, Resemble AI's $30M+ funding, and Microsoft Nuance's enterprise voice security suite).
- **Regional SAM — $320M+ (Central Asia + Turkic Markets + CIS + MENA Sovereign Voice Security):**
  - **200M+ Turkic & bilingual Slavic-Turkic speakers** (Kazakhstan, Uzbekistan, Azerbaijan, Kyrgyzstan, Turkey, CIS) plus regulated MENA banking/telecom buyers needing **non-English + sovereign on-premise** voice screening.
  - Bottom-up buyer universe across target expansion regions:
    - **450+ commercial banks & FinTechs** (incl. 21 Tier-2 banks + 150+ MFIs/payment orgs in KZ, 35 banks in UZ, 24 in AZ, 55+ in TR, plus CIS/MENA) × `$60k` avg ACV = `$27M+` (immediate Tier-1/2 KZ+CA+Turkic banking) scaling to `$110M+` across CIS/MENA banking.
    - **85+ mobile & VoIP telecom carriers** (Kcell, Beeline, Tele2/Altel, Kazakhtelecom, Ucell, Azercell, Turkcell, regional carriers) × `$220k` avg ACV = `$18.7M` (CA/Turkic) → `$95M+` regional.
    - **3,000+ enterprise contact centers & sovereign gov/defense agencies** × `$38k–$120k` avg ACV = `$115M+`.
- **Beachhead SOM (Year 1–3: Kazakhstan → Uzbekistan & Azerbaijan) — $4.2M ARR:**
  - Anchored by the **250+ institutions already connected to the NBK Anti-Fraud Center**: targeting 45 paid B2B/B2G deployments (25 banks/FinTechs at `$48k` ACV + 4 telecom operators at `$200k` ACV + 16 enterprise/gov on-premise licenses at `$135k` ACV).

---

## 4. Competitive Landscape & Defensible Moat

| Capability | **QALQAN Voice** | Pindrop ($100M) | Resemble AI ($30M) | Nuance / Microsoft |
|---|---|---|---|---|
| **Air-Gapped On-Premise (Local Data Residency)** | **Native (CPU Docker/ONNX, 0 egress)** | Cloud/hybrid US enterprise | Cloud API primary | Heavy legacy enterprise |
| **Kazakh + RU/KZ Code-Switching & Turkic Focus** | **Core specialization** | English/US call centers | Generic 50+ single-lang | English/EU focus |
| **WhatsApp / Telegram Voice Note Shield** | **Native (Opus 16–24k + Bot)** | SIP/telephony only | API only (no bot flywheel) | Telephony only |
| **CPU Inference Latency & Cost** | **19–31 ms / clip (CPU, >85% GM)** | Heavy enterprise appliance | GPU cloud inference | Heavy enterprise stack |
| **Time to Pilot & Entry Price** | **< 1 day · from $2.5k/mo** | 6–12 months · $250k+/yr | Self-serve cloud | 6–12 months · $300k+/yr |

**3 Compounding Moats:**
1. **Citizen Bot → B2B Threat-Intelligence Flywheel:** Free Telegram/WhatsApp voice screening for citizens and SMBs surfaces emerging regional scam scripts and newly deployed voice-cloning kits in the wild (using consented, anonymized acoustic embeddings — never storing PII audio), continuously improving detection before banks are hit.
2. **Sovereign Compliance-by-Design:** RAM-only processing, CPU-only ONNX deployment, and zero cross-border data transfer unlock regulated banking, telecom, and defense budgets inaccessible to US cloud APIs.
3. **50× Inference Cost Advantage:** Light CNN + acoustic physics runs at ~25 ms on a single CPU core (`< $0.0002/min` compute cost), yielding **85%+ SaaS gross margins** and making carrier-scale screening economically viable in emerging markets.

---

## 5. Business Model, Unit Economics & Go-To-Market
- **Pricing Tiers (B2B SaaS / Annual Licensing):**
  - **FinTech & Contact-Center API:** `$2,500–$8,000 / mo` (`$30k–$96k ACV`, usage-tiered per 100k screened minutes)
  - **Enterprise Bank / Sovereign On-Premise License:** `$60k–$150k ACV` (unlimited local CPU containers, SIEM/antifraud API integration, quarterly vocoder model updates)
  - **Telecom Carrier Network Screening:** `$180k–$350k ACV` (network-level call tagging on SIP/IMS gateways)
  - **Citizen Telegram / WhatsApp Bot:** Free (acquisition, brand trust & consented threat-intelligence flywheel)
- **Target Unit Economics:** Blended ACV `$65k`, Gross Margin `85%+` (CPU-only inference), target CAC `$15k–$22k` via institutional sandboxes & NBK Anti-Fraud Center ecosystem, LTV/CAC `> 8×`, Payback `< 5 months`.
- **Phased Go-To-Market:**
  - **Phase 1 (Months 1–3 — Validation & Sandbox):** Publish blinded KZ/RU/code-switching EER/FPR benchmark (Common Voice `kk`/`ru` + consented speakers + ElevenLabs/XTTS-v2/RVC); launch citizen Telegram bot publicly in Kazakhstan; run 3 structured on-premise PoCs with KZ Tier-2 banks & telecom security teams (targeting `< 1%` FPR, `< 3%` EER).
  - **Phase 2 (Months 4–12 — Paid Pilots & NBK Ecosystem):** Convert initial pilots to paid annual contracts (`$30k–$150k ACV`); integrate signal outputs with the **National Bank of Kazakhstan Anti-Fraud Center** (250+ member network).
  - **Phase 3 (Months 12–24 — Regional Turkic & CIS Expansion):** Replicate sovereign on-premise bank & carrier playbook across Uzbekistan, Azerbaijan, Kyrgyzstan, and Turkey.
