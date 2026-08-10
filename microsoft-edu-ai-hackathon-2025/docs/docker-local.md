# Běh v Dockeru (lokálně i pro sdílené použití)

Větev `docker-setup` přidává kompletní kontejnerizovaný běh aplikace — **včetně Ollamy**,
takže kdokoli s Dockerem spustí celý stack jedním příkazem a nemusí instalovat Python,
Javu, ffmpeg ani Ollamu.

**Žádné API klíče nejsou potřeba** — LLM je lokální Ollama (její OpenAI-kompatibilní
endpoint používá jen dummy klíč `ollama`). Volitelně lze v UI nastavit vlastní
OpenAI-kompatibilní endpoint (tam by klíč byl), ale pro běh to není nutné.

## Výchozí režim: vše v Dockeru

```bash
docker compose up -d --build
```

Co se stane:

1. Postaví se image backendu (Flask + gunicorn + RuleKit/Java + OpenCV + ffmpeg)
   a frontendu (Vite build servírovaný nginxem, `/api/*` proxy na backend).
2. Nastartuje kontejner `ollama` a jednorázový kontejner `ollama-pull`, který
   **automaticky stáhne model `qwen2.5vl:7b` (~6 GB)** — jen při prvním startu,
   model zůstává ve volume `ollama_data`.
3. UI běží na `http://localhost:8080`, backend API přímo na `http://localhost:5001`.

První start tedy chvíli trvá (stažení modelu); průběh sleduj přes
`docker compose logs -f ollama-pull`.

> **Výkon na macOS:** Docker na Macu nemá přístup ke GPU — inference Qwen2.5-VL 7B
> běží na CPU a je pomalá (řádově minuty na obrázek). Pro demo/test workflow to stačí;
> pro svižnou práci použij variantu s nativní Ollamou níže, nebo stack provozuj na
> linuxovém stroji s NVIDIA GPU (odkomentuj `gpus: all` u služby `ollama`
> v `docker-compose.yml`).

## Rychlejší vývojová varianta na macOS (nativní Ollama + aplikace v Dockeru)

Nativní Ollama aplikace používá Metal (GPU) a je řádově rychlejší než CPU v kontejneru:

```bash
brew install --cask ollama          # jednorázově
ollama pull qwen2.5vl:7b            # jednorázově (~6 GB)
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up -d backend frontend
```

(Spouští se jen `backend` a `frontend`; kontejnerová Ollama se nenastartuje.)

## Přepínač LLM služby přes env (`LLM_PROVIDER`)

Když nechceš čekat na 6GB model ani na CPU inferenci, přepni backend na externí
službu. **Jedna proměnná volí službu, druhá dává klíč** — nemusí být vyplněné
všechny klíče, stačí ten pro zvolenou službu. Nastav je v souboru `.env` v rootu
repa (viz `.env.example`; skutečný `.env` je v gitignore), nebo na příkazové řádce:

| `LLM_PROVIDER` | Klíč | Výchozí model (srovnatelný s qwen2.5vl:7b) |
|---|---|---|
| `ollama` (default) | žádný | `qwen2.5vl:7b` (lokální) |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` |

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up -d --build backend frontend
```

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=AIza... docker compose up -d --build backend frontend
```

Chování:

- s externí službou se spouští jen `backend` + `frontend` — služby
  `ollama`/`ollama-pull` nejsou potřeba,
- výchozí model z UI (`qwen2.5vl:7b`) se automaticky přemapuje na model zvolené
  služby; jiný model vybereš přes `LLM_MODEL=...`,
- oba provideři mají OpenAI-kompatibilní endpoint s podporou obrázků (vision);
  parametry, které daný model nepodporuje (`temperature`,
  `max_completion_tokens`), backend automaticky vynechá/nahradí,
- chybějící klíč pro zvolený provider vrátí srozumitelnou chybu přímo v UI,
- explicitní „custom LLM endpoint" nastavený v UI má stále přednost,
- pokročilé: jakoukoli jinou OpenAI-kompatibilní službu (např. OpenAI API)
  připojíš přes `LLM_BASE_URL` + `LLM_API_KEY`.

**Pozor:**

- Potřebuješ **API klíč s kreditem** (console.anthropic.com / aistudio.google.com).
  Klíč patří jen do `.env`, ne do gitu (`.env` je v gitignore).
- Média se v tomto režimu **posílají externí službě** — pro citlivá data
  (např. lékařské snímky) zvaž, zda je to přípustné; lokální Ollama drží data
  on-premises (což je pointa článku).
- Indikátor „Ollama" v UI může svítit červeně (health check kontroluje jen Ollamu),
  aplikace ale externí endpoint volá normálně.

## Ověření, že to běží

```bash
curl -s http://localhost:5001/health
# očekávané: {"ok": true, "ollama": true}   (ollama: false = model server nedostupný)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/api/health
# očekávané: 200
docker compose logs -f ollama-pull   # průběh stahování modelu
```

## Co je jinak než v plné lokální instalaci

- Backend image používá `requirements-server.txt` (bez torch / faster-whisper, stejně
  jako produkční server): **přepisy audia z videí se přeskočí** (video funguje, jen bez
  transkriptu). Obrázková pipeline je plnohodnotná.
- Data (sessions, checkpointy, uploady, stažený model) žijí v named volumes —
  přežijí `docker compose down`; smaže je až `docker compose down -v`.

## Zastavení

```bash
docker compose down          # zachová data ve volumes (vč. staženého modelu)
docker compose down -v       # smaže i data a model
```
