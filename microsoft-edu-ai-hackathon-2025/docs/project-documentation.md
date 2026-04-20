# Media Feature Lab — Technická dokumentace

## 1. Přehled systému

Media Feature Lab je webová aplikace pro automatickou klasifikaci/regresi multimediálního obsahu (videa, obrázky) na základě popisů generovaných velkými jazykovými modely (LLM). Systém implementuje kompletní ML pipeline v pěti fázích:

1. **Feature Discovery** — LLM analyzuje vzorky médií a navrhne sadu měřitelných features
2. **Feature Extraction (training)** — LLM extrahuje hodnoty features z trénovacích médií
3. **Model Training** — ML model trénovaný na extrahovaných features
4. **Feature Extraction (testing)** — Extrakce features z testovacích médií
5. **Prediction** — Predikce s evaluačními metrikami

### 1.1 Typický end-to-end průchod systémem

| Fáze | Vstup od uživatele | Co dělá backend | Hlavní výstup |
|---|---|---|---|
| 1. Discovery | Ukázková média + název cílové proměnné + target mode | LLM analyzuje omezený počet reprezentativních vzorků a navrhne strukturovanou `feature_spec` | JSON specifikace featur |
| 2. Training Extraction | Trénovací ZIP + volitelně CSV s labely | LLM extrahuje hodnoty featur pro každý soubor v datasetu | `training_X.csv`, načtené `dataset_Y`, progress job |
| 3. Training | Výběr cílového sloupce z CSV | RuleKit natrénuje model a vypočítá metriky | pravidla, metriky, feature importance |
| 4. Testing Extraction | Testovací ZIP + stejná `feature_spec` | Extrakce stejných featur z neviděných dat | `testing_X.csv` |
| 5. Prediction | Volitelně testovací `dataset_Y` pro evaluaci | Batch predikce nad `testing_X` a případné porovnání s ground truth | tabulka predikcí, evaluační metriky |

Prakticky je důležité rozlišit:

- **Discovery** neprochází celý dataset, ale jen malý reprezentativní vzorek.
- **Extraction** už naopak zpracovává všechna média v trénovacím nebo testovacím ZIPu.
- **Training** a **Prediction** jsou asynchronní joby, jejichž stav frontend pravidelně polluje.

### 1.2 Omezení a předpoklady pro interpretaci výsledků

Při psaní praktické části je vhodné explicitně uvést tato provozní omezení:

- Discovery analyzuje maximálně prvních `DISCOVERY_MAX_SAMPLES` médií (výchozí 10, konfigurovatelné env proměnnou). Slouží k návrhu featur, ne k plné analýze datasetu.
- Všechna LLM volání jsou globálně serializována přes file lock (fcntl), takže při více uživatelích nebo více paralelních jobech roste latence kvůli frontě.
- Extrakce nad větším datasetem může trvat minuty až desítky minut podle zvoleného modelu, velikosti vstupu a vytížení Ollamy.
- Párování `dataset_X` a `dataset_Y` používá normalizované basename bez přípony; tolerují se cesty, přípony, uvozovky a velikost písmen, ale názvy musí stále odpovídat stejnému médiu.
- Klasifikace validuje, že cílová proměnná je skutečně kategorická; sloupce s vysokou kardinalitou nebo téměř spojitým numerickým rozsahem jsou odmítnuty.
- Výsledky jsou interpretovatelné, ale jejich kvalita závisí na kvalitě lokálního multimodálního modelu v Ollamě a na kvalitě vstupních dat.

### Stav implementace (aktualizace 2026-04-09)

Tato sekce doplňuje historický popis níže o aktuální chování aplikace:

- Fáze 3 a Fáze 5 jsou asynchronní (`/train`, `/predict` vrací `job_id` a frontend polluje `/status/{job_id}`).
- Pipeline podporuje `target_mode`:
  - `regression`: RuleKit (`RuleRegressor`), metriky `mse`, `mae`, `correlation`.
  - `classification`: RuleKit classifier s interpretovatelnými pravidly; metriky `accuracy`, `balanced_accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `mcc`, `confusion_matrix`.
- Predikce ve classification režimu vrací `predicted_label`, `confidence`, volitelně `actual_label`.
- Predikce v regression režimu vrací `predicted_score`, volitelně `actual_score`.
- Při `target_mode=classification` se cílová proměnná validuje jako skutečně kategorická. Sloupec s vysokou kardinalitou / téměř spojitou numerickou škálou je odmítnut s chybou a doporučením přepnout na regresi.
- Škálování features bylo odstraněno dle požadavku (pipeline běží bez `StandardScaler`).
- Frontend používá lokalizaci CZ/EN s automatickou detekcí jazyka prohlížeče při prvním načtení a perzistencí volby (`localStorage`, key `mflLang`).
- EN lokalizace je napojena i na klíčové texty 5fázového wizardu (phase titles/descriptions, hlavní CTA tlačítka, continue/stop akce, completion badges).
- Runtime hlášky z `useTrainingPipeline` (fallback progress labely a frontendové error prefixy) respektují zvolený jazyk CZ/EN.
- Produkční routování API: Apache na `llmfeatures.vse.cz` proxyuje `/api/*` na backend (`llm.vse.cz:5000`).
- Discovery endpoint přijímá ZIP nebo více samostatných médií, ale pro samotný návrh featur analyzuje maximálně `DISCOVERY_MAX_SAMPLES` médií (výchozí 10).
- Frontend po refreshi obnovuje nejen Fáze 1-4, ale i uložené výsledky Fáze 5 (`predictions`, `prediction_metrics`) a umí znovu navázat na aktivní async job ve Fázích 1-5.
- Export/import relace (`/export-session`, `/import-session`) je dostupný přes stejný origin `llmfeatures.vse.cz/api/*`.
- Session registry eviktuje neaktivní session po 24 hodinách.
- Serializace Ollama volání je implementována přes globální file lock (`fcntl`) sdílený mezi procesy.
- Konstanta `MAX_CONTENT_LENGTH` (2 GB) je definována v `config.py` a importována do `app.py`.
- `speech_service.py` používá standardní `logging` místo `print()`.
- Bare `except Exception:` klauzule v `ml_training.py` logují varování místo tiše polykat chyby.

### Architektura

```
┌─────────────────────────────────────────────────┐
│                   Frontend                        │
│         React + Vite + TailwindCSS               │
│    (SPA s 5-krokovým wizard průvodcem)            │
├─────────────────────────────────────────────────┤
│    Vite Dev Proxy / Apache Reverse Proxy           │
│ /discover,/extract,/train,/predict,… → backend    │
├─────────────────────────────────────────────────┤
│                   Backend                         │
│     Flask + Gunicorn (1 worker, 8 threads)         │
│    Routes → Pipeline logic → Services            │
├─────────────────────────────────────────────────┤
│               Služby (Services)                   │
│  ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
│  │ Ollama   │ │ Whisper   │ │ OpenCV/ffmpeg  │  │
│  │ (LLM)   │ │ Large-v3  │ │ (video proc.)  │  │
│  └──────────┘ └───────────┘ └────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Technologický stack

| Komponenta | Technologie | Verze |
|---|---|---|
| Frontend | React, TypeScript, Vite, TailwindCSS | React 18, Vite 7 |
| Backend | Python, Flask, Gunicorn | Python 3.11+, Flask 3.x |
| LLM | Ollama (lokální) — Qwen 2.5 VL 7B | ollama latest |
| Whisper | faster-whisper (large-v3, GPU, float16) | - |
| ML modely | RuleKit (Java), scikit-learn | - |
| Video | OpenCV, ffmpeg | cv2 4.x |
| Process manager | PM2 | - |
| Proxy | Apache reverse proxy (`llmfeatures.vse.cz` → `llm.vse.cz:5000`) | - |

---

## 2. Backend — Detailní popis

### 2.1 Adresářová struktura

```
backend/
├── app.py                      # Flask application factory, get_pipeline() helper
├── config.py                   # Konstanty (UPLOAD_FOLDER, CHECKPOINT_FOLDER, ...)
├── jobs.py                     # Thread-safe async job registry (s TTL cleanup)
├── session_registry.py         # Per-session pipeline registry (izolace uživatelů)
├── pipeline/
│   ├── model.py                # MachineLearningPipeline — centrální stavový objekt
│   ├── feature_discovery.py    # Fáze 1: LLM-based feature specification
│   ├── feature_extraction.py   # Fáze 2/4: Multi-pass feature extraction
│   ├── feature_schema.py       # Normalizace feature_spec do strukturovaného tvaru
│   ├── feature_validation.py   # Validace a clamping extrahovaných hodnot
│   ├── ml_training.py          # Fáze 3/5: dispatcher train/predict
│   ├── ml_classification.py    # Klasifikační větev (validace cíle, metriky, CV)
│   ├── ml_regression.py        # Regresní větev (RuleKit only)
│   ├── ml_preprocessing.py     # One-hot, imputace, oversampling menšinové třídy
│   └── ml_rules.py             # Extrakce pravidel z RuleKitu + covering-rule lookup
├── routes/
│   ├── discover.py             # POST /discover
│   ├── extract.py              # POST /extract, POST /extract-local
│   ├── train.py                # POST /train
│   ├── predict.py              # POST /predict
│   ├── status.py               # GET /status/<job_id>
│   ├── reset.py                # POST /reset
│   ├── state.py                # GET /state
│   ├── health.py               # GET /health, GET /queue-info
│   └── session_transfer.py     # GET /export-session, POST /import-session
├── services/
│   ├── processing.py           # Media processing (video → keyframes + audio)
│   ├── openai_service.py       # LLM API client (Ollama OpenAI-compatible) + GPU lock
│   └── speech_service.py       # Whisper transcription (faster-whisper)
├── tests/                      # Pytest unit testy pro čisté util funkce
└── utils/
    ├── file_utils.py           # ZIP extraction, media file discovery
    ├── csv_utils.py            # CSV/labels loading utilities
    ├── target_context.py       # Resolver target sloupce napříč moduly
    ├── ollama_errors.py        # Detekce GPU load / transient Ollama chyb
    └── retry.py                # Exponential backoff retry helper
```

### 2.2 Centrální stavový objekt — `MachineLearningPipeline`

**Soubor:** `backend/pipeline/model.py`

Každý uživatel (prohlížeč) dostane vlastní instanci `MachineLearningPipeline`, identifikovanou podle session ID (viz sekce 2.9). Instance je vytvořena při prvním requestu a uložena v `session_registry`. Každá instance persistuje svůj stav do **oddělené podložky** `checkpoints/sessions/{session_id}/`, takže souběžní uživatelé si navzájem nepřepisují data.

#### Atributy

| Atribut | Typ | Popis |
|---|---|---|
| `feature_spec` | `dict` | Definice featur ve schématu `{name: [min,max]}` nebo `{name: ["cat_a","cat_b"]}` |
| `target_variable` | `str` | Název cílové proměnné |
| `training_X` | `DataFrame \| None` | Extrahované features z trénovacích dat |
| `training_Y_df` | `DataFrame \| None` | Labels CSV (ground truth) |
| `model` | `RuleRegressor \| RuleClassifier \| None` | Natrénovaný RuleKit model |
| `xgb_model` | `None` | Legacy pole; vždy `None` (XGBoost byl odstraněn) |
| `scaler` | `None` | Zachováno jen kvůli zpětné kompatibilitě; aktuální pipeline neškáluje features |
| `rules` | `list[str]` | Extrahovaná pravidla z RuleKit |
| `target_mode` | `str` | `regression` nebo `classification` |
| `training_Y_column` | `str` | Název cílového sloupce použitého při tréninku |
| `mse` | `float \| None` | RuleKit MSE na trénovacích datech |
| `rulekit_mse` | `None` | Legacy pole; vždy `None` |
| `xgb_mse` | `None` | Legacy pole; vždy `None` |
| `cv_mse`, `cv_mae`, `cv_std` | `float \| None` | Cross-val metriky pro regresi |
| `train_accuracy`, `cv_accuracy`, ... | `float \| None` | Trénovací a cross-val metriky pro klasifikaci |
| `warnings` | `list[str]` | Ne-fatal upozornění z tréninku |
| `is_trained` | `bool` | Indikátor natrénovaného modelu |
| `testing_X` | `DataFrame \| None` | Extrahované features z testovacích dat |
| `predictions` | `list[dict] \| None` | Batch výstup Fáze 5 |
| `prediction_metrics` | `dict \| None` | Vyhodnocení predikcí vůči ground truth |
| `_training_columns` | `list[str]` | Sloupce po preprocessing (pro alignment) |
| `_scaler_mean` | `list[float]` | Legacy kompatibilita; aktuálně nepoužíváno |
| `_scaler_scale` | `list[float]` | Legacy kompatibilita; aktuálně nepoužíváno |

#### Persistence

Pipeline stav se ukládá do `checkpoints/sessions/{session_id}/`:

| Soubor | Formát | Obsah |
|---|---|---|
| `pipeline_state.json` | JSON | Metadata pipeline (feature_spec, target_mode, pravidla, metriky, warnings, predictions, prediction_metrics, legacy scaler params...) |
| `training_X.csv` | CSV | Extrahované trénovací features |
| `training_Y_df.csv` | CSV | Trénovací labels |
| `testing_X.csv` | CSV | Extrahované testovací features |
| `model.pkl` | Pickle | RuleKit Java model object |
| `xgb_model.pkl` | Pickle | Legacy; vždy prázdné (XGBoost odstraněn) |

Frontend endpoint `GET /state` z těchto dat skládá hydratační payload, takže po refreshi dokáže obnovit:

- dokončené fáze 1-5
- `train_result`
- `predictions` a `prediction_metrics`
- dropdown pro výběr cílového sloupce (`dataset_Y_columns`)

**Migrační logika:** Pokud existuje legacy `pipeline_state.pkl` (starý pickle formát), systém ho automaticky načte, re-uloží v novém JSON+CSV formátu a smaže starý soubor.

---

### 2.3 Fáze 1: Feature Discovery

**Soubor:** `backend/pipeline/feature_discovery.py`

Dvouúrovňový proces pro automatický návrh feature specifikace:

#### Krok 1: Pozorování vzorků

Do discovery lze poslat ZIP i více samostatných médií. Po načtení / rozbalení backend pracuje s nalezenými mediálními soubory, ale pro samotný návrh featur záměrně analyzuje jen **maximálně prvních `DISCOVERY_MAX_SAMPLES` vzorků (výchozí 10)**. Tento krok tedy slouží k rychlému feature engineeringu, ne k plné analýze datasetu.

Pro každý z max. `DISCOVERY_MAX_SAMPLES` vzorkových médií se zavolá LLM s observation promptem:

```
You are a media analysis AI.
Carefully observe this media clip and describe what you perceive —
visual content, motion, audio characteristics, mood, pacing, people,
objects, environment, and any other notable properties.
Be objective and specific. Output a concise bullet-point list of observations.
```

**Zpracování videa:** Pro video se nejdříve extrahují keyframes (scene-based selection) + audio transcript (Whisper), tyto se přiloží k promptu jako kontext.

#### Krok 2: Syntéza feature specifikace

LLM obdrží všechny pozorování + informace o target variable a navrhne 5-8 features:

```
You are a machine learning feature engineer.
Your goal is to predict: '{target_variable}'.

Below are observations from N media sample(s):
[observations]

[labels context]

Based on these observations, define EXACTLY 5 to 8 measurable features that:
- Can be extracted from ANY media clip of this type
- Are likely to correlate with '{target_variable}'
- Cover DIVERSE perceptual dimensions (visual, audio, temporal, semantic)
- Have clear, unambiguous measurement criteria
- Are independent from each other

Output STRICTLY a JSON object with 5–8 keys.
Each value MUST be either a numeric range [min, max]
or a categorical domain ["value_a", "value_b", ...].
```

**Parametry:** Temperature 0.3 (nízká kreativita pro deterministický výstup)

**Labels context podle target mode:**
- `regression`: min/max/mean/std + ukázkové hodnoty
- `classification`: počet tříd, distribuce tříd a ukázkové labely

**JSON extrakce:** `json.JSONDecoder().raw_decode()` — najde první validní JSON objekt v odpovědi. Filtruje meta klíče (`summary`, `classification`, `reasoning`). Hard cap na 8 features.

**Normalizace schématu:** Výstup z discovery se následně převádí přes `feature_schema.normalize_feature_spec(...)` na kanonický tvar. Preferovaný a ukládaný formát je:
- numerická feature: `[min, max]`
- kategorická feature: `["hodnota_a", "hodnota_b", ...]`

Legacy stringové popisy jsou při načtení nebo při volání `/extract` pouze best-effort převedeny do tohoto tvaru kvůli zpětné kompatibilitě.

**Výstup:** `dict {feature_name: schema}`, např.:
```json
{
  "action_intensity": [0, 10],
  "speech_presence": [0, 1],
  "scene_type": ["indoor", "outdoor", "mixed"]
}
```

#### Progress reporting (aktuální implementace)

Discovery nyní reportuje jemnější progress přes `progress_cb`:

- 3-5%: příprava vstupů
- 5-60%: analýza jednotlivých vzorků (`Analyzuji vzorek i/N: file`)
- 65%: LLM syntéza feature spec
- 90%: parsování JSON odpovědi
- 100%: dokončeno

---

### 2.4 Fáze 2/4: Feature Extraction

**Soubor:** `backend/pipeline/feature_extraction.py`

Extrahuje hodnoty features z každého média podle feature_spec.

Před samotnou extrakcí se `feature_spec` vždy normalizuje do strukturovaného schématu:
- `[min, max]` pro numerické features
- `["a", "b", "c"]` pro kategorické features

To platí jak pro freshly generated spec z discovery, tak pro spec načtený ze session nebo poslaný klientem přes `/extract` a `/extract-local`.

#### Chain-of-Thought Extraction Prompt

```
You are a feature extraction AI analyzing media content.

First, briefly describe what you observe in this media
(2-3 sentences about visual content, audio, mood, pacing).

Then extract EXACTLY these features:
  - action_intensity: integer in [0, 10]
  - speech_presence: integer in [0, 1]
  - scene_type: one of ["indoor", "outdoor", "mixed"]
  ...

Context: The target variable can be either:
- regression: numeric range + mean/std for calibration
- classification: class labels + class distribution for separation cues

Output format: First your brief observation (2-3 sentences),
then on a new line output ONLY a valid JSON object with the exact keys.
Each value MUST respect the specified range or categorical domain.
```

**Proč chain-of-thought:** Nutí LLM nejdříve analyzovat obsah a teprve pak kvantifikovat. Snižuje počet "hádalých" numerických odpovědí.

#### Multi-pass extrakce

Konfigurovatelné přes env var `EXTRACTION_PASSES` (default: 1).

Pro každé médium:
1. Zavolá LLM `N`-krát se stejným promptem
2. Agregace výsledků:
   - **Numerické features:** medián z validních hodnot
   - **Kategorické features:** majority vote (nejčastější hodnota)
3. Pokud všechny passy selžou → `None` (bude imputováno)

**Účel:** LLM výstup je stochastický — jeden call může vrátit odlehlou hodnotu. Medián ze 2+ callů dramaticky snižuje varianci.

#### Feature Validation & Clamping

**Soubory:** `backend/pipeline/feature_schema.py`, `backend/pipeline/feature_validation.py`

Po každém LLM callu se extrahované hodnoty validují a clampují podle deklarovaných rozsahů ve feature_spec:

- Schema-first validace: pokud je hodnota definovaná jako `[min,max]`, výstup se clampuje numericky do rozsahu.
- Pokud je hodnota definovaná jako enum pole (`["cat_a","cat_b",...]`), výstup se ověřuje proti povoleným kategoriím.
- Legacy string popisy (`"score 0-10"`, `"binary 0 or 1"`) jsou stále podporované kvůli zpětné kompatibilitě starších session, ale před dalším zpracováním se nejdřív převádějí na strukturovaný tvar.

**Doporučený formát pro nové integrace:** neposílat stringové popisy, ale rovnou strukturované hodnoty `[min,max]` nebo enum pole.

| Popis ve feature_spec | Detekce (regex) | Akce |
|---|---|---|
| `"score 0-10"` | `(\d+)\s*[-–to]+\s*(\d+)` | clamp na [0, 10] |
| `"binary 0 or 1"` | `binary\|0\s+or\s+1\|0/1` | round na 0/1 |
| `"percentage"` | `percent\|percentage\|%` | clamp na [0, 100] |
| jinak | — | bez akce |

### 2.4.1 Přenos relace mezi servery (Export/Import session)

Pipeline je přenositelná mezi servery přes snapshot aktuální session.

1. `GET /export-session` vrátí ZIP s checkpointem session.
2. `POST /import-session` přijme ZIP, obnoví checkpoint a naváže na předchozí stav.

Frontend má pro tento workflow tlačítka **Export relace** a **Import relace** v horním panelu.

**Poznámky kompatibility:**

- Doporučená je stejná verze backendu na obou serverech.
- Server musí mít kompatibilní Python závislosti (načítání `model.pkl`).
- Pokud se interní formát checkpointu mezi verzemi změní, import může selhat.

Loguje statistiky: kolik hodnot bylo clampnuto, pro které features.

#### Imputace chybějících hodnot

Po extrakci všech řádků se chybějící hodnoty (`None`) **okamžitě** nahradí **mediánem sloupce** v rámci aktuální dávky. Tím pádem `training_X.csv` ani UI nikdy neobsahují prázdné buňky.

Proč medián, ne 0:
- 0 může být validní hodnota na okraji škály
- Medián neposouvá distribuci

**Poznámka k metodologii:** Training pipeline si každý CV fold imputuje zvlášť (z trénovací části foldu), aby nedošlo k data leakage. Okamžitá imputace v extrakci tuto logiku nenarušuje, protože pracuje s celou dávkou nezávisle na trénovacích labelech.

Pokud LLM pro dané médium nevrátí žádný validní JSON (např. poškozený soubor, timeout modelu), vznikne prázdný řádek s `media_name`. Po imputaci mediánem dostane tento řádek hodnoty typické pro daný dataset — při malém počtu takových řádků to neovlivní výsledky.

#### Checkpoint & Resume

Průběžný checkpoint: `checkpoints/extract_{dataset_type}.json`. Pokud extrakce spadne, při dalším spuštění se přeskočí již zpracovaná média. Po úspěšném dokončení se checkpoint smaže.

---

### 2.5 Zpracování médií — Video pipeline

**Soubor:** `backend/services/processing.py`

#### Pro video:

1. **Audio extrakce:** ffmpeg → MP3 (`libmp3lame`, quality 2)
2. **Transkripce:** faster-whisper (large-v3, GPU, float16, beam_size=5)
   - Vrací: `full_text`, `segments` (s timestamps), `language`
   - Truncation: max 12 000 znaků v promptu
3. **Keyframe extrakce:** Scene-change detection
   - Sampling ~200 framů rovnoměrně
   - HSV histogram pro každý frame (`cv2.calcHist` + `cv2.normalize`)
   - Bhattacharyya distance mezi po sobě jdoucími framy
   - Scene change = distance > 0.35
   - Výběr: frames u scene changes + první a poslední frame
   - Fallback: rovnoměrné doplnění pokud málo scén
   - Max 10 keyframes
4. **LLM call:** Všechny keyframes jako base64 JPEG + transcript + timestamps → Ollama API

#### Pro obrázek:

1. Konverze na base64 JPEG
2. Přímý LLM call s jedním obrázkem

#### LLM API call

**Soubor:** `backend/services/openai_service.py`

- **Endpoint:** `http://localhost:11434/v1` (Ollama OpenAI-compatible API)
- **Model:** Qwen 2.5 VL 7B (konfigurovatelné)
- **System prompt:** `"You are a feature extraction assistant. You MUST output valid JSON only."`
- **Max tokens:** 2048
- **Temperature:** 0.1
- **Retry:** 3 pokusy s exponenciálním backoff (2s → 4s → 8s)
- **Response cleaning:** Odstraňuje markdown code blocks (`` ```json ... ``` ``)
- **JSON parsing:** `json.loads(clean_content)`, fallback: raw content s error flagou

---

### 2.6 Fáze 3: Model Training

**Soubor:** `backend/pipeline/ml_training.py`

**Route:** `backend/routes/train.py` (`POST /train`) běží asynchronně a vrací `job_id`.

Train route respektuje `target_mode` zvolený na začátku pipeline a spouští odpovídající validační a modelovací větev.

#### Preprocessing

`_preprocess_features(df, training_columns=None)`:

1. Listy → comma-separated strings
2. Object (string) sloupce:
   - `.lower()` → `.strip()` → whitespace → underscores
   - Null-like hodnoty (`nan`, `not_applicable`, `n/a`, `none`) → `None`
3. One-hot encoding (`pd.get_dummies`)
4. Deduplikace sloupců
5. Alignment na `training_columns` při predikci (chybějící = 0)

#### Feature normalizace

Škálování přes `StandardScaler` bylo z pipeline odstraněno. Modely pracují přímo s výstupem `_preprocess_features(...)`, tedy s one-hot zakódovanými a zarovnanými featurami bez další numerické normalizace.

#### Regrese: RuleKit RuleRegressor

**RuleKit RuleRegressor:**
- Java-based rule induction (knihovna RuleKit)
- Pracuje na neskálovaných datech — rules jsou interpretabilní
- Produkuje lidsky čitelná pravidla typu `IF feature > threshold THEN prediction`
- Výstup: trénovací `mse` na celém datasetu

Cross-validace pro regresi není implementována (`cv_mse`, `cv_std`, `cv_mae` jsou vždy `None`).

#### Feature Importance

- **RuleKit:** Normalizovaná frekvence features v pravidlech. Hodnota = podíl pravidel, v jejichž podmínkách se daná feature vyskytuje (rozsah 0–1, součet ≤ 1). Formát: `{"feature_name": 0.42, ...}` seřazeno sestupně. Implementace: `_count_rule_features()` v `ml_rules.py`.

Vrací se v API response pro vizualizaci ve frontend.

#### Klasifikace: validace cíle a model

Při `target_mode = classification` backend nejdřív validuje, že zvolený target sloupec opravdu vypadá jako kategorická proměnná:
- musí obsahovat alespoň 2 různé třídy
- nesmí být prázdný po odfiltrování chybějících hodnot
- pokud je téměř celý numerický a má vysokou kardinalitu, je považován za pravděpodobně spojitou proměnnou a backend vrátí chybu s doporučením přepnout na regresi
- pokud má vysokou kardinalitu i jako string a většina tříd je singleton nebo téměř singleton, je považován za identifier-like sloupec a backend ho odmítne

**Positive label pro binární klasifikaci** — prioritní pořadí v `_resolve_positive_label()`:
1. Env var `CLASSIFICATION_POSITIVE_LABEL` (explicitní override)
2. Třída pojmenovaná `1`, `true`, `positive`, `malignant`, nebo `yes`
3. Fallback: méně četná třída

Model pro klasifikaci:
- `RuleClassifier` z RuleKitu
- finální klasifikační výstup vzniká přímo z RuleKitu
- pravidla zůstávají viditelná i v klasifikačním režimu
- pokud RuleKit vrátí pravděpodobnosti, frontend zobrazuje i confidence
- `StratifiedKFold` místo obyčejného `KFold`, aby foldy zachovávaly rozložení tříd

Metriky klasifikace:
- train: `accuracy`, `balanced_accuracy`, `f1_macro`, `mcc`
- cross-validation: `cv_accuracy`, `cv_balanced_accuracy`, `cv_f1_macro`, `cv_precision_macro`, `cv_recall_macro`, `cv_mcc`

V klasifikační větvi se nepoužívá `MSE` ani `MAE`.

#### Async progress fáze (aktuální implementace)

`train_model(..., progress_cb=...)` publikuje průběh po hlavních krocích:

- 10%: příprava tréninku
- 15%: příprava features
- 25%: trénink RuleKit
- 60%: RuleKit predikce
- 80%: K-fold cross-validation (pouze klasifikace)
- 98%: ukládání modelu/checkpointu
- 100%: dokončeno

#### Výstup tréninku (v `status.details` po dokončení jobu)

```json
{
  "status": "success",
  "mse": 0.0234,                    // RuleKit MSE (training)
  "rules_count": 12,
  "rules": ["IF action_intensity > 5.5 AND speech_presence = 1 THEN 0.72", ...],
  "feature_importance": {
    "rulekit": {"action_intensity": 0.42, "speech_presence": 0.33, ...}
  },
  "warnings": [],
  "training_data_X": [...]          // Extrahovaná data pro UI
}
```

---

### 2.7 Fáze 5: Prediction

**Soubor:** `backend/pipeline/ml_training.py` → `predict_batch()`

**Route:** `backend/routes/predict.py` (`POST /predict`) běží asynchronně a vrací `job_id`.

1. Preprocessing testovacích features (alignment na training columns)
2. Imputace mediánů (stejné hodnoty jako při tréninku)
3. RuleKit predikce (`predict()`)
4. Pokud jsou k dispozici actual labels: MSE, MAE, Pearson correlation

#### Klasifikační predikce

Pokud je pipeline v režimu klasifikace:
1. provede se preprocessing features stejně jako při tréninku
2. použije se `RuleClassifier`
3. výsledný label vzniká přímo z RuleKitu; confidence je dostupná, pokud ji RuleKit vrátí
4. frontend zároveň zobrazuje použité RuleKit pravidlo, pokud se podaří dohledat pokrývající rule
5. pokud jsou k dispozici ground-truth labels, musí CSV obsahovat stejný target sloupec jako při tréninku; jinak backend vrátí chybu místo tichého fallbacku na jiný sloupec
6. pokud jsou k dispozici ground-truth labels, počítají se klasifikační metriky:
   - `accuracy`
   - `balanced_accuracy`
   - `f1_macro`
   - `precision_macro`
   - `recall_macro`
   - `mcc`
   - `confusion_matrix`
   - per-class metriky (`precision`, `recall`, `f1`, `support`)
   - průměrná confidence, confidence pro správné a chybné predikce

V klasifikační větvi se nepočítá `MSE` ani `MAE`.

Frontend v přehledu predikcí navíc zvýrazňuje klasifikační řádky přímo v tabulce:
- zeleně správně predikované položky
- červeně chybné predikce
- žlutě položky bez ground-truth labelu

#### Async progress fáze (aktuální implementace)

`predict_batch(..., progress_cb=...)` publikuje průběh po hlavních krocích:

- 5-10%: příprava + preprocessing testovacích features
- 25%: RuleKit predikce
- 60-88%: sestavení výsledků po položkách (`Sestavuji výsledky i/N`)
- 92%: výpočet evaluačních metrik
- 100%: dokončeno

---

### 2.8 Asynchronní zpracování

**Soubor:** `backend/jobs.py`

Thread-safe registr pro async joby:

```python
_lock = threading.Lock()
_registry: dict = {}

def set_job(job_id, value): ...
def get(job_id): ...
def update_job(job_id, **kwargs): ...
```

**Workflow:**
1. Klient pošle POST request (discover/extract/train/predict)
2. Server vytvoří `job_id = uuid4()`, vrátí `{"job_id": "..."}`
3. Spustí zpracování v background threadu
4. Klient polluje `GET /status/{job_id}` pro progress updates
5. Odpověď: `{"progress": 45, "stage": "Extracting (3/10)...", "done": false}`
6. Po dokončení: `{"progress": 100, "done": true, "details": {...}}`

**TTL dokončených jobů:** Dokončené joby se mažou z paměti **8 hodin po posledním update** (pole `_updated_at`). Po vypršení TTL vrátí `GET /status/{job_id}` chybu `404`. Výsledky jsou uloženy na disku v `pipeline_state.json` — ztráta `job_id` neznamená ztrátu výsledků. Frontend po `404` ze `/status` automaticky přejde na obnovu stavu ze `/state`.

---

### 2.9 Víceuživatelská podpora a Ollama serializace

#### Session izolace

Původní architektura používala jediný globální `pipeline` singleton — všechny HTTP requesty sdílely tentýž objekt. Při souběžném použití dvěma uživateli docházelo k přepisování stavu (např. `training_X`, `rules`, checkpoint soubory).

**Řešení:** Per-session pipeline registry (`backend/session_registry.py`).

```
prohlížeč A ──X-Session-ID: abc──► get_pipeline("abc") ──► MachineLearningPipeline A
                                                                └── checkpoints/sessions/abc/
prohlížeč B ──X-Session-ID: xyz──► get_pipeline("xyz") ──► MachineLearningPipeline B
                                                                └── checkpoints/sessions/xyz/
```

Session ID je UUID generovaný prohlížečem při první návštěvě a uložen v `localStorage`. Posílá se jako HTTP hlavička `X-Session-ID` s každým requestem. Server identifikuje uživatele a vrátí (nebo vytvoří) jeho pipeline instanci.

Session registry eviktuje neaktivní session po 24 hodinách (TTL-based cleanup). Checkpoint složka na disku (`checkpoints/sessions/{id}/`) zůstane — smaže se pouze in-memory instance pipeline.

**Cascade invalidace při opakovaném spuštění fáze** (`pipeline.invalidate_from_phase(n)`):

| Spuštěná fáze | Co se vymaže |
|---|---|
| 1 (Discovery) | `feature_spec`, `training_X/Y`, model, `testing_X`, predikce |
| 2 (Training Extraction) | `training_X/Y`, model, `testing_X`, predikce |
| 3 (Training) | model, predikce |
| 4 (Testing Extraction) | `testing_X`, predikce |
| 5 (Prediction) | pouze predikce |

#### Ollama EOF — popis chyby a oprava

**Chybová zpráva:**
```
Error code: 500 - {'error': {'message': 'do load request: Post "http://127.0.0.1:40125/load": EOF'}}
```

**Příčina:** Ollama používá interní HTTP server pro správu modelů. Při prvním requestu je model načten (`/load`). Pokud ve stejnou chvíli přijde druhý request (jiný uživatel nebo dvě otevřená okna), oba se snaží model načíst souběžně. Vnitřní Ollama server vrátí druhému requestu `EOF` — spojení bylo uzavřeno protože model byl již v procesu načítání.

Konkrétně k chybě dochází, pokud:
1. Uživatel A spustí Fázi 1 (Feature Discovery) → background thread čeká na Ollama
2. Uživatel B (nebo druhá záložka) spustí jakoukoliv fázi → další Ollama request

**Oprava:** Globální file lock přes `fcntl.flock(...)` v `backend/services/openai_service.py` serializuje všechna volání Ollama API i napříč procesy. Druhý thread/proces čeká, dokud první nedokončí svůj LLM call.

```python
with open(_OLLAMA_LOCK_FILE, "w") as lockfile:
  fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
  response = local_client.chat.completions.create(...)
  fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
```

**Dopad na výkon:** Extrakce více médií jedním uživatelem probíhá sekvenčně (bylo tak i dříve). Souběžní uživatelé se řadí do fronty na úrovni LLM callů. Celková doba zpracování je stejná, ale bez crashů.

#### CPU fallback při GPU OOM

Pokud Ollama selže při načítání modelu s GPU out-of-memory chybou, backend automaticky zopakuje request v CPU režimu (`num_gpu=0`). Chování:
- Povoleno výchozí: `OLLAMA_CPU_FALLBACK=1`
- Selhání GPU se zaloguje jako `WARNING` v server logu
- CPU inference je výrazně pomalejší (desítky sekund per médium místo sekund)
- Nastavením `OLLAMA_CPU_FALLBACK=0` lze fallback zakázat, čímž GPU OOM skončí chybou extrakce

---

## 3. Frontend — Detailní popis

### 3.1 Architektura

React SPA s TypeScript, build systém Vite, styling TailwindCSS.

```
frontend/src/
├── App.tsx                         # Root shell aplikace, skládá header + TrainingView
├── hooks/
│   ├── useTrainingPipeline.ts      # Hlavní orchestrace 5fázového pipeline stavu
│   ├── usePollProgress.ts          # Generic polling helper pro async joby
│   ├── usePipelineRuntime.ts       # Health check + queue polling
│   ├── trainingPipelineRequests.ts # API requesty a job polling wrappery
│   ├── trainingPipelineState.ts    # Invalidation / reset lokálního pipeline stavu
│   ├── trainingPipelineRecovery.ts # Restore state + persist helpery po reloadu
│   ├── trainingPipelineUtils.ts    # localStorage persistence + textové konstanty
│   ├── useAppUi.ts                 # Theme / jazyk / guide preference
│   └── useSessionTransfer.ts       # Export / import relace
├── components/
│   ├── AppHeader.tsx               # Horní lišta aplikace a globální akce
│   ├── TrainingView.tsx            # Tenký wrapper nad 5 fázemi wizardu
│   ├── training-view/
│   │   ├── DiscoveryPhasePanel.tsx
│   │   ├── TrainingExtractionPhasePanel.tsx
│   │   ├── TrainingPhasePanel.tsx
│   │   ├── TestingExtractionPhasePanel.tsx
│   │   ├── PredictionPhasePanel.tsx
│   │   ├── PredictionResults.tsx
│   │   ├── TrainingResultsCard.tsx
│   │   ├── FeatureSpecBox.tsx
│   │   ├── DatasetTable.tsx
│   │   ├── FileDropZone.tsx
│   │   ├── ProgressBar.tsx
│   │   ├── QueueBusyBanner.tsx
│   │   ├── OllamaWarning.tsx
│   │   ├── progressHooks.ts
│   │   ├── errorHelpers.ts
│   │   ├── style.ts
│   │   ├── translations.ts
│   │   └── shared.tsx             # Barrel re-export nad training-view moduly
│   ├── Guide.tsx                  # Úvodní nápověda / onboarding
│   └── ui/                        # Reusable UI primitives
└── lib/
    ├── api.ts                      # API URL konstanty, typy, session header helper
    ├── helpers.tsx                 # Obecné frontend utility (typy souborů, transcripty…)
    └── pipelineDownloads.ts        # Export/download funkcionalita pipeline artefaktů
```

### 3.2 Custom Hook — `useTrainingPipeline`

**Soubor:** `frontend/src/hooks/useTrainingPipeline.ts`

Hook po refaktoru funguje jako tenká orchestrace nad menšími specializovanými moduly:

- drží reaktivní stav všech 5 fází (`useState` pro busy flagy, data, výsledky, progress)
- deleguje API requesty a polling do `trainingPipelineRequests.ts`
- deleguje health/queue runtime logiku do `usePipelineRuntime.ts`
- deleguje reset/invalidation do `trainingPipelineState.ts`
- deleguje restore/persist helpery do `trainingPipelineRecovery.ts` a `trainingPipelineUtils.ts`
- používá `localStorage` jen pro lightweight metadata (step, target variable, mode, feature spec, model provider), nikoli pro velké datasety
- po reloadu obnovuje stav z `GET /state` a navazuje i na aktivní async job (`discover`, `extract_training`, `extract_testing`, `train`, `predict`)
- po dokončené Fázi 5 obnovuje i `predictions` a `prediction_metrics`, takže výsledky predikce se po refreshi neztratí

Tím je logika lépe testovatelná a odděluje:
- stavový orchestration layer
- network/request layer
- persistence/recovery layer
- runtime monitoring layer

#### Předvýběr cílového sloupce

Ve Fázi 3 se po načtení `datasetYColumns` předvybírá **poslední** sloupec z CSV (typicky score/label), nikoliv první ID sloupec.

### 3.3 Polling mechanismus

**Soubor:** `frontend/src/hooks/usePollProgress.ts`

```typescript
pollProgress(jobId, onStatus, abortSignal)
```

Polluje `GET /status/{jobId}` každých N ms, volá callback s aktuálním stavem. Podporuje abort (zrušení uživatelem).

Při síťové chybě používá exponenciální backoff (od cca 600 ms do 5 s) a po úspěšném pollu delay resetuje.

Vyšší vrstvy (`trainingPipelineRequests.ts`) tento low-level polling obalují do pojmenovaných funkcí:
- `pollDiscoveryJob(...)`
- `pollExtractJob(...)`
- `pollTrainJob(...)`
- `pollPredictJob(...)`

Tím frontend nepracuje přímo s raw polling helperem v každé fázi zvlášť.

### 3.4 Frontend wizard komponenty

Původní velký `TrainingView.tsx` byl rozdělen do menších komponent podle odpovědnosti:

- `TrainingView.tsx`:
  - phase stepper
  - error banner
  - queue banner
  - model selector
  - předávání props do jednotlivých phase panelů
- `DiscoveryPhasePanel.tsx` až `PredictionPhasePanel.tsx`:
  - každý soubor odpovídá jedné fázi wizardu
- `PredictionResults.tsx` a `TrainingResultsCard.tsx`:
  - výsledkové panely oddělené od vstupních formulářů
- malé UI stavebnice (`FeatureSpecBox`, `DatasetTable`, `FileDropZone`, `ProgressBar`) jsou samostatné moduly

Refaktor odstranil původní monolitický render strom a zlepšil:
- čitelnost
- možnost izolovaných úprav jednotlivých fází
- opětovné použití komponent
- orientaci v UI logice během dalšího vývoje

### 3.5 API komunikace a session identifikace

Vite dev server proxyuje API requesty na backend:
```
/discover → http://127.0.0.1:5000/discover
/extract  → http://127.0.0.1:5000/extract
...
```

#### Apache reverse proxy (produkční nasazení)

Na `llmfeatures.vse.cz` běží Apache, který:

- servíruje statický frontend build z `/var/www/llmfeatures/current`
- proxyuje `/api/*` na backend server `http://llm.vse.cz:5000/`
- HTTPS je ukončené na Apache (Certbot/Let's Encrypt)
- prohlížeč komunikuje pouze se stejným originem `llmfeatures.vse.cz`, backend nemusí být veřejně dostupný

Konfigurace: [docs/apache-llmfeatures.vse.cz.conf](apache-llmfeatures.vse.cz.conf).

#### X-Session-ID

Každý request přidává hlavičku `X-Session-ID` s UUID uloženým v `localStorage` prohlížeče. UUID se generuje automaticky při první návštěvě. Tímto mechanismem backend rozliší uživatele bez nutnosti přihlášení.

```typescript
// frontend/src/lib/api.ts
export function sessionHeaders(): Record<string, string> {
  return { "X-Session-ID": getSessionId() };  // UUID z localStorage
}

// Použití v každém fetch call:
fetch(TRAIN_URL, {
  method: "POST",
  headers: { ...sessionHeaders(), "Content-Type": "application/json" },
  body: JSON.stringify({ target_column }),
});
```

---

## 4. Klíčové algoritmy a rozhodnutí

### 4.1 Proč LLM-based features?

Tradiční přístupy ke klasifikaci videa používají hand-crafted features (HOG, SIFT, optical flow) nebo deep learning embeddings (CNN, ViT). Tento systém zkouší alternativní přístup:

1. LLM pozoruje médium a generuje přirozený textový popis
2. Z popisu se extrahují strukturované features (numerické/kategorické)
3. Features se použijí pro klasický interpretovatelný ML model (RuleKit)

**Výhody:**
- Interpretabilita — features i pravidla jsou lidsky čitelná
- Flexibilita — stejný systém funguje pro jakýkoliv typ média a cílovou proměnnou
- Automatický feature engineering — LLM navrhuje features relevantní pro daný úkol

**Nevýhody:**
- Stochastičnost — LLM vrací různé hodnoty pro stejný vstup (řešeno multi-pass + medián)
- Latence — každé médium vyžaduje 1-3 LLM cally (mitigováno async zpracováním)
- Závislost na kvalitě LLM — menší modely (7B) mají omezenou přesnost

### 4.2 Proč scene-based keyframes?

Rovnoměrný sampling keyframů může produkovat redundantní obrázky (8 framů z jedné scény). Scene-change detection:

1. Samplinguje ~200 framů z videa
2. Počítá HSV histogram pro každý
3. Bhattacharyya distance mezi po sobě jdoucími framy detekuje změny scény
4. Vybírá frame z každé unikátní scény

Výsledek: LLM vidí maximálně diverzní vizuální informaci z videa.

### 4.3 Proč RuleKit pro oba režimy?

RuleKit je zvolen jako jediný ML model pro klasifikaci i regresi:
- produkuje lidsky čitelná pravidla (`IF feature > threshold THEN prediction`)
- uživatel vidí konkrétní pravidlo, které zdůvodnilo každou predikci
- u bakalářské práce je interpretovatelnost pipeline klíčová
- backend i frontend zůstávají jednoduché a udržovatelné

### 4.4 Feature Validation & Clamping

LLM občas generuje hodnoty mimo deklarovaný rozsah (např. "score 0-10" ale vrátí 15). Validační pipeline:

1. Parsuje expected range z feature_spec description (regex)
2. Clampuje numerické hodnoty do deklarovaného rozsahu
3. Zaokrouhluje binary features na 0/1
4. Loguje statistiky pro analýzu spolehlivosti LLM

---

## 5. Možná rozšíření — Computed Features (baseline experiment)

### 5.1 Koncept

Kromě LLM-generovaných features je možné extrahovat deterministické (computed) features přímo z mediálních souborů pomocí OpenCV a ffmpeg. Tyto features poskytují spolehlivý baseline pro porovnání s LLM-based přístupem.

### 5.2 Navrhované computed features

#### Video features

| Feature | Popis | Metoda výpočtu |
|---|---|---|
| `duration_seconds` | Délka videa v sekundách | `cv2.CAP_PROP_FRAME_COUNT / cv2.CAP_PROP_FPS` |
| `fps` | Framerate | `cv2.CAP_PROP_FPS` |
| `resolution_pixels` | Rozlišení (width × height) | `cv2.CAP_PROP_FRAME_WIDTH * cv2.CAP_PROP_FRAME_HEIGHT` |
| `avg_brightness` | Průměrný jas přes keyframes | Konverze na grayscale, `np.mean()` přes všechny keyframe pixely |
| `color_saturation` | Průměrná saturace v HSV | `cv2.cvtColor(HSV)`, průměr S kanálu |
| `motion_intensity` | Intenzita pohybu | `cv2.calcOpticalFlowFarneback()` mezi po sobě jdoucími framy, průměrná magnitude optického flow vektoru |
| `scene_cut_count` | Počet střihů/změn scény | Počet peaků v histogram distance (Bhattacharyya > threshold) |
| `shot_frequency` | Frekvence střihů | `scene_cut_count / duration_seconds` |
| `audio_energy_rms` | RMS energie audio stopy | ffmpeg loudnorm filter nebo pydub `.rms` |
| `speech_ratio` | Poměr řeči ku celkové délce | Součet délek Whisper segmentů / celková délka videa |

#### Obrázek features

| Feature | Popis | Metoda výpočtu |
|---|---|---|
| `resolution_pixels` | Rozlišení | width × height |
| `avg_brightness` | Průměrný jas | Grayscale mean |
| `color_saturation` | Saturace | HSV S-kanál mean |
| `edge_density` | Hustota hran | `cv2.Canny()`, poměr edge pixelů ku celkovému počtu |
| `color_diversity` | Barevná diverzita | Počet unikátních barev v kvantizovaném RGB histogramu (32 binů per kanál) |

### 5.3 Implementační návrh

```python
# backend/pipeline/computed_features.py

def compute_hard_features(media_path: str, whisper_segments: list | None = None) -> dict:
    """Compute deterministic features from media file.

    Returns dict with computed features. Never fails — returns 0 for errors.
    """
    features = {}

    if is_video(media_path):
        cap = cv2.VideoCapture(media_path)
        features["duration_seconds"] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        features["fps"] = cap.get(cv2.CAP_PROP_FPS)
        features["resolution_pixels"] = (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Keyframe-based features
        frames = extract_keyframes(media_path)
        features["avg_brightness"] = np.mean([
            np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames
        ])
        features["color_saturation"] = np.mean([
            np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:,:,1]) for f in frames
        ])

        # Optical flow
        if len(frames) >= 2:
            flows = []
            for i in range(1, len(frames)):
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flows.append(np.mean(magnitude))
            features["motion_intensity"] = np.mean(flows)

        # Speech ratio
        if whisper_segments:
            speech_duration = sum(s["end"] - s["start"] for s in whisper_segments)
            features["speech_ratio"] = speech_duration / features["duration_seconds"]

    return features
```

### 5.4 Experimentální design pro BP

Pro srovnávací experiment v bakalářské práci doporučuji následující design:

1. **LLM-only model:** Trénovat pouze na features extrahovaných LLM (stávající pipeline)
2. **Computed-only model:** Trénovat pouze na computed features (baseline)
3. **Combined model:** Trénovat na spojení LLM + computed features
4. **Evaluace:** Pro každý model reportovat MSE, MAE, Pearson correlation + K-fold CV

**Očekávané výsledky:**
- Computed features zachytí low-level signály (jas, pohyb, délka) — dobrý baseline
- LLM features zachytí high-level sémantiku (nálada, téma, narativní struktura)
- Combined model by měl dosáhnout nejlepších výsledků
- Klíčový přínos BP: demonstrace, že LLM popisy přidávají informaci nad rámec low-level signálů

---

## 6. Konfigurace a nasazení

### 6.1 Environment variables

| Proměnná | Default | Popis |
|---|---|---|
| `VITE_API_BASE` | `""` (prázdný — požadavky jdou na origin stránky) | Frontend API base URL; v dev režimu Vite proxy přesměruje `/api/*` na localhost:5000 |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,https://llmfeatures.vse.cz` | CORS allowlist backendu; čárkami oddělené originy — musí obsahovat doménu frontendu |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL lokálního Ollama serveru |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | Výchozí Ollama model pro discovery/extrakci |
| `OLLAMA_API_KEY` | `ollama` | API klíč pro OpenAI-kompatibilní endpoint Ollamy (ve většině instalací stačí default) |
| `OLLAMA_NUM_CTX` | `4096` | Context window Ollamy (tokeny) |
| `OLLAMA_REQUEST_TIMEOUT` | `120.0` | HTTP request timeout proti Ollamě (sekundy) |
| `OLLAMA_CONNECT_TIMEOUT` | `5.0` | HTTP connect timeout proti Ollamě (sekundy) |
| `OLLAMA_CPU_FALLBACK` | `1` | Povolit CPU fallback (`num_gpu=0`) při selhání GPU loadu |
| `FLASK_DEBUG` | `0` | Debug mode (0 = production) |
| `PORT` | `5000` | Port, na kterém poslouchá backend |
| `DISCOVERY_MAX_SAMPLES` | `10` | Počet vzorků analyzovaných LLM při Phase 1 — vyšší znamená bohatší feature spec, ale delší discovery |
| `EXTRACTION_PASSES` | `1` | Počet LLM callů per médium při extrakci (více = stabilnější hodnoty, vyšší cena) |
| `CV_MAX_FOLDS` | `3` | Horní limit počtu CV foldů |
| `CLASSIFICATION_POSITIVE_LABEL` | — (neuvedeno) | Explicitní positive label pro binární klasifikaci; jinak se odvozuje heuristikou ze jmen tříd (`positive`, `yes`, `true`, `1`) nebo z méně četné třídy |
| `CLASSIFICATION_POSITIVE_THRESHOLD` | `0.45` | Práh pro positive prediction v binární klasifikaci |
| `SECRET_KEY` | náhodný `token_hex(32)` při každém startu | Flask session secret — pro stabilitu sessions mezi restarty nastavit jako perzistentní hodnotu |

**Poznámka k PM2 konfiguraci:** `EXTRACTION_PASSES` a `CV_MAX_FOLDS` jsou nastaveny přímo v `backend/ecosystem.config.js` (sekce `env`). Při použití PM2 tedy jejich hodnota vychází z tohoto souboru, nikoliv z `backend/.env`. Změna hodnot vyžaduje úpravu `ecosystem.config.js` a restart procesů přes `pm2 reload ecosystem.config.js`.

**Poznámka k ALLOWED_ORIGINS:** Musí obsahovat veřejný frontend origin, jinak polling na `/status/<job_id>` a `/queue-info` skončí s CORS chybou 403.

**Poznámka k SECRET_KEY:** Pokud není nastaveno, Flask generuje nový náhodný klíč při každém restartu — session cookies přestanou platit. V produkci nastavit jako perzistentní hodnotu.

Poznámka: backend načítá konfiguraci explicitně z `backend/.env`; kořenový `.env` se nepoužívá.

Pro přesun backendu na jiný server viz [docs/migration.md](../docs/migration.md) a bootstrap script [scripts/bootstrap_backend.sh](../scripts/bootstrap_backend.sh).

### 6.2 PM2 procesy

| Proces | Příkaz | Port |
|---|---|---|
| `backend` | `gunicorn -w 1 -b 0.0.0.0:5000 --timeout 1200 --graceful-timeout 60 app:app` | 5000 |

### 6.3 Spuštění

```bash
cd backend
pm2 start ecosystem.config.js
```

Pro split deploy variantu s frontendem na samostatném Apache hostu viz [frontend-llmfeatures-deploy.md](frontend-llmfeatures-deploy.md) a konfiguraci [apache-llmfeatures.vse.cz.conf](apache-llmfeatures.vse.cz.conf).

### 6.4 Reprodukovatelný experimentální scénář pro BP

Pro praktickou část bakalářské práce je vhodné popsat alespoň jeden standardizovaný experiment:

1. Připravit trénovací ZIP a testovací ZIP se stejným typem médií.
2. Připravit CSV s cílovou proměnnou, kde první sloupec odpovídá názvům médií.
3. Ve Fázi 1 spustit discovery a nechat vygenerovat `feature_spec`.
4. Ve Fázi 2 extrahovat `training_X` z trénovacího datasetu.
5. Ve Fázi 3 vybrat cílový sloupec a natrénovat model.
6. Ve Fázi 4 extrahovat `testing_X` ze separátního testovacího datasetu.
7. Ve Fázi 5 spustit predikci a případně přiložit testovací `dataset_Y` pro evaluaci.
8. Exportovat artefakty (`feature_spec`, `training_X`, `testing_X`, `predictions`, `rules`, `metrics`) pro další analýzu a screenshoty do práce.

Tento scénář je dobře přenositelný do textu praktické části, protože přímo kopíruje strukturu uživatelského rozhraní i backendové pipeline.

---

## 7. API Reference

### Podporované formáty vstupu

| Typ | Přípony |
|---|---|
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` |
| Obrázek | `.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.gif` |
| Archiv | `.zip` |

Limity: maximální velikost uploadu je **2 GB** (Flask `MAX_CONTENT_LENGTH`). ZIP archiv nesmí po rozbalení překročit **5 GB** (kontrola před extrakcí).

---

### POST /discover
Spustí feature discovery z vzorkových médií.

**Request:** `multipart/form-data`
- `files`: Mediální soubory nebo ZIP archivy se vzorky
- `target_variable`: Název cílové proměnné
- `model`: ID LLM modelu
- `labels_file` (optional): CSV s labels

Poznámka: endpoint může přijmout více souborů nebo ZIP, ale discovery pro samotný návrh featur analyzuje maximálně prvních `DISCOVERY_MAX_SAMPLES` nalezených médií (výchozí 10).

**Response:** `{"job_id": "uuid"}`

### POST /extract
Spustí feature extraction ze ZIP archivu.

**Request:** `multipart/form-data`
- `file`: ZIP archiv s médii
- `model`: ID LLM modelu
- `feature_spec`: JSON string se strukturovanými definicemi featur
- `dataset_type`: `"training"` nebo `"testing"`
- `labels_file` (optional): CSV s labels

**Doporučený tvar `feature_spec`:**
```json
{
  "action_intensity": [0, 10],
  "speech_presence": [0, 1],
  "scene_type": ["indoor", "outdoor", "mixed"]
}
```

**Response:** `{"job_id": "uuid"}`

### POST /extract-local
Extraction z lokální cesty na serveru.

**Request:** `application/json`
```json
{
  "zip_path": "/path/to/data.zip",
  "model": "qwen2.5vl:7b",
  "feature_spec": {
    "action_intensity": [0, 10],
    "scene_type": ["indoor", "outdoor", "mixed"]
  },
  "dataset_type": "training",
  "labels_path": "/path/to/labels.csv"
}
```

**Response:** `{"job_id": "uuid"}`

### GET /status/{job_id}
Vrátí stav async jobu.

**Response:**
```json
{
  "progress": 45,
  "stage": "Extracting (3/10): video_001.mp4 [pass 1/2]",
  "done": false
}
```

### GET /state
Vrátí hydratační snapshot aktuální session pro obnovu frontendu po refreshi.

**Response (zkráceně):**
```json
{
  "feature_spec": {...},
  "target_variable": "memorability_score",
  "target_mode": "regression",
  "completed_phases": [1, 2, 3, 4, 5],
  "suggested_step": 5,
  "training_data_X": [...],
  "testing_data_X": [...],
  "dataset_Y_columns": ["media_name", "label"],
  "train_result": {...},
  "predictions": [...],
  "prediction_metrics": {...}
}
```

### POST /train
Spustí trénování modelu.

**Request:** `application/json`
```json
{"target_column": "memorability_score"}
```

**Response (okamžitě):**
```json
{"job_id": "uuid"}
```

Finální výsledek je dostupný přes `GET /status/{job_id}` v `details` (viz sekce 2.6).

**Regresní výsledek:**
```json
{
  "status": "success", "target_mode": "regression",
  "mse": 0.023, "rules_count": 12,
  "rules": ["IF action_intensity > 5.5 THEN 0.72"],
  "feature_importance": {"rulekit": {"action_intensity": 0.42}},
  "warnings": [], "training_data_X": [...]
}
```

**Klasifikační výsledek:**
```json
{
  "status": "success", "target_mode": "classification",
  "train_accuracy": 0.85, "train_balanced_accuracy": 0.83,
  "train_f1_macro": 0.82, "train_mcc": 0.78,
  "cv_accuracy": 0.79, "cv_balanced_accuracy": 0.77,
  "cv_f1_macro": 0.76, "cv_precision_macro": 0.78,
  "cv_recall_macro": 0.77, "cv_mcc": 0.71, "cv_folds": 5,
  "rules_count": 8, "rules": ["IF ..."],
  "feature_importance": {"rulekit": {"feat": 0.33}},
  "warnings": [], "training_data_X": [...]
}
```

### POST /predict
Spustí batch predikci.

**Request:** `multipart/form-data` (optional `labels_file`) nebo prázdný POST.

**Response (okamžitě):**
```json
{"job_id": "uuid"}
```

Finální výsledek je dostupný přes `GET /status/{job_id}` v `details`.

**Regresní predikce:**
```json
{
  "predictions": [
    {
      "media_name": "video_001",
      "predicted_score": 0.672, "actual_score": 0.71,
      "rule_applied": "IF action_intensity > 5.5 THEN 0.672",
      "extracted_features": {...}
    }
  ],
  "metrics": {
    "mode": "regression",
    "mse": 0.0234, "mae": 0.112, "correlation": 0.89,
    "matched_count": 50, "total_count": 50
  }
}
```

**Klasifikační predikce:**
```json
{
  "predictions": [
    {
      "media_name": "video_001",
      "predicted_label": "high", "confidence": 0.82,
      "actual_label": "high",
      "rule_applied": "IF ...",
      "extracted_features": {...}
    }
  ],
  "metrics": {
    "mode": "classification",
    "accuracy": 0.75, "balanced_accuracy": 0.73,
    "f1_macro": 0.72, "precision_macro": 0.74, "recall_macro": 0.73,
    "mcc": 0.67,
    "confusion_matrix": [[30, 10], [5, 55]],
    "per_class_metrics": {
      "high": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 40}
    },
    "avg_confidence": 0.79, "correct_confidence_avg": 0.84,
    "wrong_confidence_avg": 0.71,
    "matched_count": 100, "total_count": 100
  }
}
```

### POST /reset
Vymaže všechny checkpointy a resetuje pipeline stav.

**Response:** `{"ok": true, "removed_checkpoints": [...]}`

### GET /health
Vrátí základní healthcheck backendu a dostupnost Ollamy.

**Response:**
```json
{
  "ok": true,
  "ollama": true
}
```

### GET /queue-info
Vrátí informaci o vytížení inference fronty pro Ollamu.

**Response:**
```json
{
  "busy": true,
  "queued": 2
}
```

Poznámka: `queued` znamená počet requestů čekajících na LLM lock, nikoliv přesný počet „jiných uživatelů“.

### GET /export-session
Exportuje aktuální session checkpoint jako ZIP archiv.

**Response:** binární ZIP soubor s checkpointy aktuální session.

### POST /import-session
Obnoví session z dříve exportovaného ZIP archivu.

**Request:** `multipart/form-data`
- `file`: ZIP soubor vytvořený přes `/export-session`

**Response:**
```json
{
  "ok": true,
  "imported_files": ["pipeline_state.json", "training_X.csv", "model.pkl"]
}
```

---

## 8. Uživatelské testování pro BP

Tato sekce slouží jako podklad pro praktickou část bakalářské práce. Je napsána tak, aby ji bylo možné snadno převzít do kapitoly o evaluaci použitelnosti aplikace i bez finálních výsledků testování.

### 8.1 Cíl uživatelského testování

Cílem uživatelského testování není ověřovat přesnost strojového učení jako takového, ale zejména:

- ověřit srozumitelnost 5fázového workflow aplikace,
- ověřit, zda uživatel dokáže samostatně projít celou pipeline od nahrání dat až po predikci,
- identifikovat matoucí nebo problematická místa rozhraní,
- ověřit, zda uživatel rozumí základním výstupům modelu a evaluačním metrikám.

Pro praktickou část práce je tento typ evaluace vhodný zejména proto, že odpovídá hlavnímu cíli aplikace: umožnit uživateli pohodlně pracovat s multimodální ML pipeline bez nutnosti programování.

### 8.2 Doporučený návrh testu

Doporučený formát testování:

- typ testu: krátké nemoderované scénářové testování,
- délka jednoho testu: přibližně 10 až 15 minut,
- počet respondentů: přibližně 5 až 8,
- forma sběru odpovědí: online dotazník po dokončení práce s aplikací.

Nemoderované testování je vhodné v situaci, kdy autor není přítomen u každého respondenta. Zároveň je organizačně jednoduché a dostatečné pro ověření základní použitelnosti systému.

### 8.3 Materiály pro testování

Pro testování je vhodné respondentům poskytnout:

- veřejný odkaz na aplikaci,
- připravený trénovací dataset ve formátu ZIP,
- připravený testovací dataset ve formátu ZIP,
- stručné zadání úkolů,
- odkaz na hodnoticí dotazník.

Pro krátké testování je doporučeno použít menší dataset, který nezatěžuje respondenta dlouhým čekáním. Prakticky se osvědčuje:

- trénovací dataset v řádu desítek až nižších stovek obrázků,
- testovací dataset v řádu desítek obrázků,
- předem určený cílový sloupec v CSV, aby se minimalizovala nejednoznačnost zadání.

Pokud je cílem především ověřit použitelnost rozhraní, je vhodnější menší a rychlejší dataset než rozsáhlý dataset, který by test zbytečně prodlužoval.

### 8.4 Doporučený scénář úkolů pro respondenty

Respondentům lze zadat následující posloupnost kroků:

1. Otevřete aplikaci a nahrajte ukázková nebo připravená data pro Fázi 1.
2. Ve Fázi 1 spusťte Discovery.
3. Ve Fázi 2 nahrajte trénovací dataset a spusťte extrakci.
4. Ve Fázi 3 vyberte určený cílový sloupec a spusťte trénink modelu.
5. Ve Fázi 4 nahrajte testovací dataset a spusťte testovací extrakci.
6. Ve Fázi 5 spusťte predikci a zobrazte výsledky.
7. Prohlédněte si výsledné metriky a tabulku predikcí.
8. Po dokončení práce s aplikací vyplňte krátký dotazník.

Tento scénář odpovídá reálnému workflow aplikace a zároveň pokrývá všechny klíčové obrazovky, které mají být z hlediska použitelnosti ověřeny.

### 8.5 Sledované ukazatele

Při vyhodnocení testování je vhodné sledovat kombinaci objektivních a subjektivních ukazatelů.

Objektivní ukazatele:

- zda respondent dokončil všechny zadané úkoly,
- přibližná doba dokončení testu,
- ve které fázi došlo k nejčastějším problémům,
- zda respondent zvládl najít a interpretovat výstupy modelu.

Subjektivní ukazatele:

- srozumitelnost rozhraní,
- logická návaznost jednotlivých fází,
- pochopení rozdílu mezi regresí a klasifikací,
- pochopení metrik a predikčních výsledků,
- schopnost používat aplikaci bez další pomoci.

Tyto ukazatele lze pohodlně sbírat přes krátký formulář typu Google Forms nebo Microsoft Forms.

### 8.6 Doporučená struktura dotazníku

Dotazník může být rozdělen do čtyř částí:

1. základní informace o respondentovi,
2. dokončení úkolů,
3. Likertovo škálové hodnocení použitelnosti,
4. otevřené otázky.

Doporučené okruhy otázek:

- vztah respondenta k technologiím nebo informatice,
- předchozí zkušenost s machine learningem nebo datovou analýzou,
- odhad času potřebného pro dokončení testu,
- nejproblematičtější fáze aplikace,
- srozumitelnost rozhraní,
- logická návaznost workflow,
- pochopení metrik a tabulky predikcí,
- návrhy na zlepšení.

### 8.7 Hotový text metodiky do bakalářské práce

Následující text lze přímo použít jako základ kapitoly o metodice uživatelského testování:

> Cílem uživatelského testování bylo ověřit použitelnost a srozumitelnost webové aplikace Media Feature Lab z pohledu běžného uživatele. Testování nebylo zaměřeno na přesnost modelu samotného, ale především na schopnost uživatele samostatně projít celou pětifázovou pipeline aplikace, orientovat se v jednotlivých krocích a porozumět výsledným výstupům.
>
> Testování bylo navrženo jako krátké nemoderované scénářové testování. Respondenti obdrželi odkaz na aplikaci, připravená testovací data a stručné zadání úkolů. Po dokončení práce s aplikací vyplnili online dotazník. Tento přístup byl zvolen s ohledem na nízkou organizační náročnost a snahu minimalizovat časové zatížení respondentů.
>
> Každý respondent pracoval se stejnou sadou dat a plnil stejnou posloupnost kroků odpovídající hlavnímu workflow aplikace. Konkrétně šlo o spuštění discovery fáze, extrakci featur z trénovacích dat, natrénování modelu, extrakci testovacích featur a spuštění predikce. Součástí testování bylo také základní ověření schopnosti uživatele interpretovat výsledné metriky a orientovat se v tabulce predikcí.
>
> Pro testování byla připravena datová sada obsahující trénovací a testovací obrázky ve formátu ZIP. Cílový sloupec pro trénování modelu byl určen předem, aby bylo možné soustředit pozornost především na použitelnost rozhraní a ne na rozhodování nad daty. Délka jednoho testu byla navržena přibližně na 10 až 15 minut.
>
> Po dokončení práce s aplikací respondenti vyplnili dotazník zaměřený na subjektivní hodnocení použitelnosti systému. Hodnocena byla zejména srozumitelnost rozhraní, návaznost jednotlivých fází, pochopení rozdílu mezi regresí a klasifikací, interpretace metrik a celková schopnost používat aplikaci bez další pomoci. Součástí dotazníku byly i otevřené otázky pro zachycení nejasností a návrhů na zlepšení.

### 8.8 Šablona pro doplnění výsledků po dokončení testování

Po získání odpovědí lze navázat následující textovou šablonou:

> Uživatelského testování se zúčastnilo celkem **[N] respondentů**. Z toho **[X] respondentů** dokončilo všechny zadané úkoly, zatímco **[Y] respondentů** uvedlo, že dokončilo pouze část testu. Průměrná nebo nejčastěji uváděná délka testu byla **[doplňte]**.
>
> Jako nejproblematičtější se ukázala zejména **[doplňte fázi nebo typ problému]**. Respondenti nejčastěji zmiňovali **[doplňte konkrétní nejasnosti]**. Naopak jako srozumitelné byly hodnoceny zejména **[doplňte]**.
>
> Z odpovědí v Likertově škále vyplynulo, že aplikace byla celkově vnímána jako **[doplňte: srozumitelná / spíše srozumitelná / problematická]**. Respondenti pozitivně hodnotili zejména **[doplňte]**, zatímco prostor pro zlepšení se ukázal v oblasti **[doplňte]**.
>
> Na základě testování byly identifikovány konkrétní návrhy na zlepšení uživatelského rozhraní, zejména **[doplňte]**. Tato zjištění slouží jako podklad pro další iteraci aplikace a zároveň potvrzují, že zvolený koncept vícefázového rozhraní je pro uživatele do značné míry pochopitelný.

### 8.9 Doporučení pro interpretaci výsledků v BP

Při psaní závěrečné interpretace je vhodné zdůraznit:

- že testování ověřovalo především použitelnost a srozumitelnost systému,
- že se jednalo o krátké scénářové testování, nikoliv o rozsáhlou UX studii,
- že cílem nebylo statisticky reprezentativní srovnání, ale identifikace hlavních problémových míst,
- že i menší počet respondentů je pro odhalení hlavních UX problémů u tohoto typu aplikace dostatečně informativní.

Takový způsob interpretace je pro bakalářskou práci metodicky obhajitelný a odpovídá rozsahu studentského prototypu.

---

## 9. Backend services v detailu

Sekce 2 dokumentovala celkovou pipeline; tato sekce doplňuje popis tří klíčových servisních modulů, které se stejně dotýkají více fází a mají vlastní provozní charakteristiky.

### 9.1 `services/openai_service.py` — Ollama klient a GPU serializace

Modul je jediné místo, kde backend volá Ollama API. Vše ostatní (discovery, extraction) používá tento wrapper.

**Hlavní odpovědnosti:**

- OpenAI-kompatibilní klient nad lokální Ollamou (`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`)
- Timeouty: `OLLAMA_CONNECT_TIMEOUT` (default 5 s) a `OLLAMA_REQUEST_TIMEOUT` (default 120 s) — oddělené, aby krátká síťová nedostupnost nezatížila stejnou čekací dobou jako dlouhý LLM call
- Context window: `OLLAMA_NUM_CTX` (default 4096 tokenů) se posílá v `options.num_ctx`
- CPU fallback: pokud `OLLAMA_CPU_FALLBACK=1` a GPU selže, retry se provede s `options.num_gpu=0`
- Global file lock (`fcntl.flock`) přes soubor `_OLLAMA_LOCK_FILE` serializuje všechny LLM cally napříč Gunicorn workery i vlákny (viz sekce 2.9)
- Detekce transient chyb: deleguje na `utils.ollama_errors.is_transient_ollama_error` (EOF, `load request`, connection refused) a `is_gpu_load_error` (GPU OOM)

**API:**

```python
with ollama_exclusive():
    response = local_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[...],
        response_format={"type": "json_object"},
    )
```

`ollama_exclusive()` je context manager, který drží file lock po celou dobu LLM callu. Nedostaneme se do stavu, kdy dva procesy chtějí načíst stejný model v tentýž okamžik (ollama EOF).

### 9.2 `services/processing.py` — zpracování médií

Abstrakce nad zpracováním multimediálních souborů pro LLM extrakci.

**Odpovědnosti:**

- Vyextrahuje upload ZIP do `UPLOAD_FOLDER/<session_id>/`
- U videí vybere keyframes přes scene detection (`scenedetect` knihovna) nebo fixní interval jako fallback
- Audio track extrahuje přes `ffmpeg` do WAV a předá `speech_service`
- Obrázky se posílají LLM jako `image_url` content part
- Každý mediální soubor transformuje na strukturu `{"keyframes": [...], "transcript": "..."}` konzumovanou fází Discovery i Extraction

Výstup je deterministický: stejný vstupní soubor → stejné keyframes → stejný prompt pro LLM. Díky tomu jsou extrakční pokusy porovnatelné.

### 9.3 `services/speech_service.py` — Whisper STT

Wrapper nad `faster-whisper`.

- Lazy-inicializovaný model (načte se při prvním volání)
- Vstup: cesta k audio souboru (WAV)
- Výstup: `{"text": "...", "segments": [...]}`
- Rekurzivně ošetřuje prázdný audio track (např. video bez zvuku) — vrací prázdný transcript, nikoliv exception
- Podporované formáty vstupu: jakýkoli formát, který zvládne předchozí `ffmpeg` demux (dle instalace)

---

## 10. ML subsystém v detailu

Po refaktoru byl původní monolitický `ml_training.py` rozdělen do pěti modulů. Sekce 2.6 popisuje algoritmické detaily; tato sekce popisuje strukturu souborů.

### 10.1 `ml_training.py` — dispatcher

Tenká orchestrační vrstva:

- `train_model(pipeline, target_column, target_mode, progress_cb=...)` — volá klasifikační nebo regresní větev podle `target_mode`
- `predict_batch(pipeline, testing_df, progress_cb=...)` — stejný dispatcher pro predikci
- `_make_progress_cb(...)` — helper pro publikaci progressu do `jobs` registru
- Konstanty: `CLASSIFICATION_POSITIVE_THRESHOLD = 0.45`

Neobsahuje trénovací logiku. Pokud se mění preprocessing nebo metriky, editují se níže uvedené moduly.

### 10.2 `ml_classification.py` — klasifikační větev

- `_validate_classification_target(df, column)` — odmítne numerické vysokokardinalitní sloupce (pravděpodobně regrese), ID-like sloupce (většina tříd = singleton) a prázdné cíle
- `_resolve_positive_label(classes)` — určí positive label (env `CLASSIFICATION_POSITIVE_LABEL` > heuristika `positive/yes/true/1` > méně četná třída)
- `_compute_classification_metrics(y_true, y_pred, y_proba)` — vrací accuracy, balanced_accuracy, f1_macro, precision_macro, recall_macro, mcc, confusion_matrix, per-class breakdown
- `_run_cross_validation(X, y, model_factory, n_folds)` — `StratifiedKFold`, aggregátor CV metrik

### 10.3 `ml_regression.py` — regresní větev

- `_train_regression_branch(...)` — fit RuleKit `RuleRegressor`; žádný ensemble ani CV
- Feature importance: frekvence features v pravidlech (RuleKit)
- Vrací `mse` na trénovacích datech; `cv_mse` je vždy `None` (regresní CV není implementováno)

### 10.4 `ml_preprocessing.py` — feature pipeline

- Median imputer pro numerické featury (zachová distribuci lépe než mean)
- One-hot encoding kategoriálních featur (`pd.get_dummies` s `drop_first=False`)
- `_oversample_minority(X, y, strategy)` — pro silně nevyvážené klasifikační úlohy; prahy nastaveny tak, aby se neaplikovalo zbytečně na mírnou nerovnováhu
- Alignment testovacích features na training columns (chybějící = 0, přebývající se dropnou)

### 10.5 `ml_rules.py` — interpretabilita pravidel

- `_extract_rules(model)` — převede RuleKit Java objekt na seznam lidsky čitelných stringů
- `_count_rule_features(rules)` — frekvence každé featury v conditions pravidel (alternativní feature importance)
- `_find_covering_rule(rules, feature_row)` — pro každou predikci najde pravidlo, které řádek pokrývá; zobrazuje se ve frontend sloupci `rule_applied`

---

## 11. Utility moduly

Malé, testovatelné funkce používané napříč backendem. Jsou pokryty unit testy (viz sekce 13).

### 11.1 `utils/csv_utils.py`

- `normalize_media_name(name)` — normalizuje název média (strip, lowercase, odstranění přípony), používá se při párování CSV labels s extrahovanými featurami
- `load_labels_csv(path)` — robustní CSV loader (automatická detekce separátoru, ošetření BOM)

### 11.2 `utils/target_context.py`

- `find_target_column(df, requested)` — case-insensitive resolver; pokud uživatel uvede `"Score"` a CSV má `"score"`, najde ho
- `build_labels_context(df, target_column)` — vrátí `{media_name → label}` dict s normalizovanými klíči (používá `normalize_media_name`)
- Používá se v train i predict větvi, aby párování extraction ↔ labels bylo deterministické

### 11.3 `utils/ollama_errors.py`

Konsoliduje detekční predikáty, které byly dříve duplikované v `openai_service.py` i `feature_discovery.py`:

- `is_gpu_load_error(exc)` — `True` pro chyby typu GPU OOM, `load request`, konkrétní EOF patterns
- `is_transient_ollama_error(exc)` — nadmnožina zahrnující i dočasnou nedostupnost (connection refused)

Tento modul je jediné správné místo, kde se rozhoduje, jestli chybu retry-ovat.

### 11.4 `utils/retry.py`

- `retry_with_backoff(fn, *, max_attempts=3, base_delay=5.0, max_delay=60.0, should_retry=None, on_retry=None)`
- Exponenciální doubling (5 s → 10 s → 20 s, …) s horním stropem `max_delay`
- `should_retry` predikát (default `None` = retry na každou `Exception`); typicky se předá `is_gpu_load_error`
- `on_retry(attempt, delay, exc)` callback pro logování
- Po vyčerpání `max_attempts` propaguje poslední výjimku s kontextem

### 11.5 `utils/file_utils.py`

ZIP extrakce a iterace médií v uploadu — bez speciálního chování hodného detailní dokumentace.

---

## 12. Error handling a retry strategie

Tato sekce shrnuje, jak backend řeší chyby napříč pipeline.

### 12.1 Transient Ollama chyby

Detekce: `utils.ollama_errors.is_gpu_load_error` / `is_transient_ollama_error`.

Příklady chyb považovaných za transient:

- `do load request: ... EOF` — souběžný load stejného modelu
- `CUDA out of memory`, `GPU load failed` — dočasné přetížení VRAM
- `connection refused` — Ollama právě startuje / restartuje

**Strategie:**

1. `ollama_exclusive()` lock zabrání většině collision chyb
2. Zbylé transient chyby se retryujou přes `retry_with_backoff(..., should_retry=is_gpu_load_error)`
3. Po `max_attempts=3` se výjimka propaguje nahoru — job selže s chybovou hláškou ve `status.details.error`
4. CPU fallback (`OLLAMA_CPU_FALLBACK=1`) poskytuje degradovaný režim místo plného selhání

### 12.2 Strukturované logování

Každý modul používá `logging.getLogger(__name__)`. Místo tichého `except Exception: pass` se používá `logger.exception(...)`, aby byla traceback dohledatelná.

Log levely:

- `DEBUG` — per-médium progress, prompty
- `INFO` — start/konec fáze, session ID, target column
- `WARNING` — overfitting warning, retry s backoffem
- `ERROR` / `EXCEPTION` — hard failure s traceback

V produkci (Gunicorn + PM2) se logy posílají do `pm2 logs backend`.

### 12.3 Frontend error propagace

Backend `/status/<job_id>` vrací `{"done": true, "error": "..."}` při selhání jobu. Frontend:

- Zobrazí banner s chybou (`errorHelpers.ts` formátuje user-friendly hlášku)
- Zachová vstupní data ve fázi, aby uživatel mohl zkusit znovu bez nahrávání
- Pro transient Ollama chyby nabízí explicit retry button (nevolá `/reset`)

### 12.4 State recovery po reloadu

Pokud uživatel provede F5 během async jobu:

1. Frontend načte `GET /state` → získá aktuální `step`, `target_column`, `feature_spec`, atd.
2. Pokud state obsahuje `active_job_id`, frontend se napojí na stávající polling (neuresetuje progress)
3. Po dokončení jobu se UI vrátí do správného kroku pipeline

Tento mechanismus je robustní vůči restartu backendu, pokud checkpoint soubor (`checkpoints/sessions/<session_id>/state.json`) existuje.

---

## 13. Vývoj a testování

### 13.1 Lokální dev setup

Viz kořenový [README.md](../README.md) — obsahuje quick-start pro backend, frontend i Ollamu.

### 13.2 Pytest — unit testy util modulů

Konfigurace:

- `backend/pytest.ini` — základní nastavení (test paths, collect pattern)
- `backend/requirements-dev.txt` — `pytest>=7.4` (instaluje se zvlášť od `requirements-server.txt`)
- `backend/tests/conftest.py` — přidává backend root do `sys.path`, aby testy viděly `utils/`, `pipeline/`, atd.

Spuštění:

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

### 13.3 Pokrytí testy

| Test soubor | Testovaný modul | Počet testů |
|---|---|---|
| `tests/test_csv_utils.py` | `utils.csv_utils.normalize_media_name` | 9 |
| `tests/test_feature_schema.py` | `pipeline.feature_schema.normalize_feature_spec` | 12 |
| `tests/test_target_context.py` | `utils.target_context.find_target_column`, `build_labels_context` | 11 |
| `tests/test_retry.py` | `utils.retry.retry_with_backoff` | 8 |

Testy jsou úmyslně omezené na čisté (deterministické, side-effect-free) funkce. Integrační testy s Ollamou a RuleKitem by vyžadovaly reálný runtime a jsou mimo scope.

### 13.4 Manuální smoke test před nasazením

1. `cd backend && python -c "from app import create_app; create_app()"` — import sanity
2. `cd backend && pytest` — unit testy
3. `cd backend && flask --app app run --port 5000` + `curl http://localhost:5000/health` — liveness
4. `cd frontend && npm run build` — TypeScript + Vite build bez chyb
5. `cd frontend && npm run dev` — projít celou 5-fázovou pipeline na malém datasetu
6. F5 během aktivní fáze — ověřit state recovery

### 13.5 Rozšíření testů — co stojí za úvahu

- `pipeline/feature_validation.py` — clamping numerických hodnot do `[min, max]` rozsahu je čistá funkce
- `pipeline/ml_preprocessing._align_columns(...)` — alignment training vs. testing columns
- `utils/ollama_errors` — predikáty na pre-kanonizované exception payloady

Frontend testy (Vitest, RTL) nejsou v repozitáři nastavené; byla by to samostatná iterace.
