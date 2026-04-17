# Media Feature Lab

Webová aplikace pro automatickou klasifikaci a regresi multimediálního obsahu (videa, obrázky) pomocí kombinace LLM-řízeného objevování featur a interpretovatelných ML modelů. Uživatel nahraje malý vzorek médií s popisky, systém navrhne měřitelné featury, extrahuje je, natrénuje ensemble RuleKit + XGBoost a dokáže predikovat na novém datasetu — včetně interpretace, které pravidlo každou predikci zdůvodnilo.

Projekt vznikl na Microsoft × VŠE Edu AI Hackathonu 2025 a dále se rozvíjí jako součást bakalářské práce.

## Tech stack

| Vrstva | Technologie |
|---|---|
| Frontend | React 18 + TypeScript, Vite, TailwindCSS, Radix UI |
| Backend | Python 3.11+, Flask, Gunicorn (1 worker, 8 threads) |
| LLM | Lokální Ollama (Qwen 2.5 VL 7B) + Whisper pro STT |
| ML | RuleKit (JPype/Java), XGBoost, scikit-learn |
| Proxy | Cloudflare Worker (frontend `llmfeatures.vse.cz`) |

## Prerekvizity

- **Python** 3.11 nebo novější
- **Node.js** 18+ (doporučeno 20+)
- **Java** JRE/JDK (RuleKit běží přes JPype na JVM)
- **FFmpeg** v `$PATH` (extrakce audia pro Whisper)
- **Ollama** s nataženým vision modelem (`ollama pull qwen2.5vl:7b`)
- ~8 GB VRAM pro GPU inferenci nebo ochota tolerovat CPU fallback

## Lokální vývoj — quick start

```bash
# 1) Naklonuj repo
cd microsoft-edu-ai-hackathon-2025

# 2) Backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-server.txt
cp .env.example .env             # edituj dle potřeby
flask --app app run --port 5000  # nebo: gunicorn app:app -w 1 --threads 8 -k gthread

# 3) Frontend (v novém terminálu)
cd ../frontend
npm install
cp .env.example .env             # VITE_API_BASE=http://localhost:5000
npm run dev                      # http://localhost:5173

# 4) Ollama (samostatný proces)
ollama pull qwen2.5vl:7b
ollama serve                     # http://localhost:11434
```

Otevři `http://localhost:5173` a projdi pipeline: Discovery → Training Extraction → Training → Testing Extraction → Prediction.

## Struktura repa

| Adresář | Obsah |
|---|---|
| [backend/](microsoft-edu-ai-hackathon-2025/backend/) | Flask API, ML pipeline, Ollama/Whisper integrace |
| [frontend/](microsoft-edu-ai-hackathon-2025/frontend/) | React SPA, 5-fázová UI pipeline, lokalizace CZ/EN |
| [docs/](microsoft-edu-ai-hackathon-2025/docs/) | Detailní technická dokumentace, deployment guides |
| [scripts/](microsoft-edu-ai-hackathon-2025/scripts/) | Bootstrap a deploy skripty |

Backend layout:
- `app.py` — Flask factory, CORS, blueprint registrace
- `pipeline/` — ML logika (discovery, extraction, training, prediction, classification, regression, preprocessing, rules)
- `routes/` — HTTP endpointy (`/discover`, `/extract`, `/train`, `/predict`, `/status`, `/state`, `/reset`, `/health`, session transfer)
- `services/` — Ollama klient + GPU lock, Whisper STT, zpracování médií
- `utils/` — CSV/labels, target-column resolver, Ollama chybové predikáty, retry helper
- `tests/` — pytest unit testy pro čisté util funkce

## Testy

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

Testy pokrývají `utils/csv_utils.py`, `pipeline/feature_schema.py`, `utils/target_context.py` a `utils/retry.py`. Integrace s Ollamou / RuleKitem není testována (vyžadovala by reálný runtime).

## Dokumentace

- [docs/project-documentation.md](microsoft-edu-ai-hackathon-2025/docs/project-documentation.md) — kompletní technická dokumentace (5 fází, API, konfigurace, ML detaily)
- [docs/migration.md](microsoft-edu-ai-hackathon-2025/docs/migration.md) — split backend/frontend na oddělené stroje
- [docs/frontend-llmfeatures-deploy.md](microsoft-edu-ai-hackathon-2025/docs/frontend-llmfeatures-deploy.md) — nasazení frontendu za Apache + Certbot
- [docs/apache-llmfeatures.vse.cz.conf](microsoft-edu-ai-hackathon-2025/docs/apache-llmfeatures.vse.cz.conf) — šablona Apache VirtualHostu

## Produkční nasazení

Viz [docs/migration.md](microsoft-edu-ai-hackathon-2025/docs/migration.md). Ve zkratce:

- Backend běží na stroji s GPU pod Gunicornem (1 worker, 8 threads, gthread, timeout 1200 s)
- Frontend je statický build servírovaný přes Apache na `llmfeatures.vse.cz`
- Cloudflare Worker směruje `/discover`, `/extract`, `/train`, `/predict`, `/status/*`, `/state`, `/queue-info`, `/health` na backend tunel
- Sessions se izolují přes `X-Session-ID` header
