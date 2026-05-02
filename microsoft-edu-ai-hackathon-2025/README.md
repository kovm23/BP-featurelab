# Media Feature Lab

Webová aplikace pro automatickou klasifikaci a regresi multimediálního obsahu (videa, obrázky) pomocí kombinace LLM-řízeného objevování featur a hybridních ML modelů. Uživatel nahraje malý vzorek médií s popisky, systém navrhne měřitelné featury, extrahuje je, natrénuje model a dokáže predikovat na novém datasetu — včetně interpretace, které pravidlo každou predikci zdůvodnilo.

V klasifikačním režimu se predikce tvoří **3-way soft vote**: RuleKit (1/3) + Random Forest (1/3) + Gradient Boosting (1/3). RuleKit zůstává zdrojem interpretovatelných pravidel zobrazených v UI, zatímco RF a GBT kompenzují jeho horší generalizaci na malých datasetech.

Projekt vznikl na Microsoft × VŠE Edu AI Hackathonu 2025 a dále se rozvíjí jako součást bakalářské práce.

## Tech stack

| Vrstva | Technologie |
|---|---|
| Frontend | React 18 + TypeScript, Vite, TailwindCSS, Radix UI |
| Backend | Python 3.11+, Flask, Gunicorn (1 worker, 8 threads) |
| LLM | Lokální Ollama (Qwen 2.5 VL 7B) + Whisper pro STT |
| ML | RuleKit (JPype/Java), scikit-learn |
| Proxy | Apache reverse proxy (`llmfeatures.vse.cz` → backend `llm.vse.cz:5000`) |

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

# 2) Backend (venv je ve složce projektu, ne v backend/)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env   # edituj dle potřeby
cd backend
flask --app app run --port 5000   # nebo: gunicorn app:app -w 1 --threads 8 -k gthread

# 3) Frontend (v novém terminálu)
cd frontend
npm install
cp .env.example .env              # VITE_API_BASE=http://localhost:5000
npm run dev                       # http://localhost:5173

# 4) Ollama (samostatný proces)
ollama pull qwen2.5vl:7b
ollama serve                      # http://localhost:11434
```

Otevři `http://localhost:5173` a projdi pipeline: Discovery → Training Extraction → Training → Testing Extraction → Prediction.

## Nové funkce (verze pro obhajobu BP)

### 3-way soft vote (klasifikace)

Klasifikační model byl upgradován z čistého RuleKitu na **hybridní 3-way soft vote**:

| Model | Podíl | Role |
|---|---|---|
| RuleKit | 1/3 | Interpretovatelná pravidla, zobrazená v UI |
| Random Forest (200 stromů) | 1/3 | Robustní ensemble, odolný vůči přetrénování |
| Gradient Boosting (100 stromů) | 1/3 | Silný na tabulárních datech, kompenzuje slabiny RuleKitu |

Finální predikce = průměr pravděpodobnostních distribucí všech tří modelů (soft voting). Cross-validace trénuje všechny tři modely v každém foldu, takže CV metriky přesně odpovídají chování na testovacích datech.

Regresní větev zůstává beze změny (RuleKit only).



### Majority class baseline

Výsledky klasifikace nyní automaticky zobrazují hodnotu **majority-class baseline** (= přesnost modelu, který vždy predikuje nejčetnější třídu). Zobrazuje se:
- v metrikovém panelu vedle accuracy: `baseline: X.X% (+Xpp)`
- v rozbalitelné sekci *Advanced metrics* ve výsledcích trénování

Nemusíš nic konfigurovat — funguje automaticky po spuštění Fáze 3 a Fáze 5.

### Export matice záměn jako PNG

V rozbalitelné sekci *Confusion Matrix* (Fáze 5) přibyl tlačítko **Download as PNG**. Kliknutím stáhneš `confusion_matrix.png` — matplotlib heatmap vhodný pro přímé vložení do přílohy BP.

Vyžaduje: `matplotlib` musí být nainstalován (`pip install -r requirements.txt` to zajistí).

### Test reprodukovatelnosti LLM (Repeatability Test)

Po úspěšné extrakci tréninkových dat (Fáze 2) se zobrazí tlačítko **LLM Repeatability Test**.

1. Nahraj jedno video/obrázek
2. Nastav počet opakování (2–10, doporučeno 3–5)
3. Klikni *Start Test* — systém zavolá LLM N-krát na stejný soubor
4. Výsledek zobrazí tabulku: pro každou featuru **mean ± std** a **CV%** (coefficient of variation)
   - CV% < 10 % → zelená (nízká variabilita)
   - CV% 10–20 % → žlutá
   - CV% > 20 % → červená

Tato čísla lze přímo citovat v bakalářské práci jako doklad míry stochasticity LLM extrakce.

### Demo mód (pro screenshoty do BP)

Umožňuje zobrazit kompletní UI se vzorkovými výsledky bez spuštěného Ollamy.

**Spuštění:**

```bash
# Backend
DEMO_MODE=true flask --app app run --port 5000

# Frontend
VITE_DEMO_MODE=true npm run dev
```

Po spuštění se v záhlaví aplikace zobrazí oranžové tlačítko **Demo**. Kliknutím se načte předpřipravená demo relace (MediaEval dataset, 4 třídy, 24 tréninkových + 10 testovacích videí). Všechny fáze pipeline se zobrazí jako dokončené s realistickými hodnotami odpovídajícími výsledkům popsaným v BP.

Demo mód je **pouze pro screenshoty** — model není reálně natrénován a predikce nelze znovu spustit.

## Struktura repa

| Adresář | Obsah |
|---|---|
| [backend/](backend/) | Flask API, ML pipeline, Ollama/Whisper integrace |
| [frontend/](frontend/) | React SPA, 5-fázová UI pipeline, lokalizace CZ/EN |
| [docs/](docs/) | Detailní technická dokumentace, deployment guides |
| [scripts/](scripts/) | Bootstrap a deploy skripty |

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

- [docs/project-documentation.md](docs/project-documentation.md) — kompletní technická dokumentace (5 fází, API, konfigurace, ML detaily)
- [docs/migration.md](docs/migration.md) — split backend/frontend na oddělené stroje
- [docs/frontend-llmfeatures-deploy.md](docs/frontend-llmfeatures-deploy.md) — nasazení frontendu za Apache + Certbot
- [docs/apache-llmfeatures.vse.cz.conf](docs/apache-llmfeatures.vse.cz.conf) — šablona Apache VirtualHostu

## Produkční nasazení

Viz [docs/migration.md](docs/migration.md). Ve zkratce:

- Backend běží na `llm.vse.cz` (GPU server s Ollamou) pod Gunicornem (1 worker, 8 threads, gthread, timeout 1200 s)
- Frontend je statický build servírovaný přes Apache na `llmfeatures.vse.cz`
- Apache proxyuje `/api/*` na `http://llm.vse.cz:5000/` — backend nemusí být veřejně dostupný
- Sessions se izolují přes `X-Session-ID` header
