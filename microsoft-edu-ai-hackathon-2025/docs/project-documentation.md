# Media Feature Lab — Technická dokumentace

## 1. Přehled systému

Media Feature Lab je webová aplikace pro automatickou klasifikaci/regresi multimediálního obsahu (videa, obrázky) na základě popisů generovaných velkými jazykovými modely (LLM). Systém implementuje kompletní ML pipeline v pěti fázích:

1. **Feature Discovery** — LLM analyzuje vzorky médií a navrhne sadu měřitelných features
2. **Feature Extraction (training)** — LLM extrahuje hodnoty features z trénovacích médií
3. **Model Training** — Ensemble RuleKit + XGBoost trénovaný na extrahovaných features
4. **Feature Extraction (testing)** — Extrakce features z testovacích médií
5. **Prediction** — Ensemble predikce s evaluačními metrikami

### Stav implementace (aktualizace 2026-04-02)

Tato sekce doplňuje historický popis níže o aktuální chování aplikace:

- Fáze 3 a Fáze 5 jsou asynchronní (`/train`, `/predict` vrací `job_id` a frontend polluje `/status/{job_id}`).
- Pipeline podporuje `target_mode`:
  - `regression`: RuleKit + XGBoost ensemble, metriky `mse`, `mae`, `correlation`.
- `classification`: RuleKit classifier s interpretovatelnými pravidly, metriky `accuracy`, `balanced_accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `mcc`, `confusion_matrix`.
- Predikce ve classification režimu vrací `predicted_label`, `confidence`, volitelně `actual_label`.
- Predikce v regression režimu vrací `predicted_score`, volitelně `actual_score`.
- Při `target_mode=classification` se cílová proměnná validuje jako skutečně kategorická. Sloupec s vysokou kardinalitou / téměř spojitou numerickou škálou je odmítnut s chybou a doporučením přepnout na regresi.
- Škálování features bylo odstraněno dle požadavku (pipeline běží bez `StandardScaler`).
- Frontend používá lokalizaci CZ/EN s automatickou detekcí jazyka prohlížeče při prvním načtení a perzistencí volby (`localStorage`, key `mflLang`).
- EN lokalizace je napojena i na klíčové texty 5fázového wizardu (phase titles/descriptions, hlavní CTA tlačítka, continue/stop akce, completion badges).
- Runtime hlášky z `useTrainingPipeline` (fallback progress labely a frontendové error prefixy) respektují zvolený jazyk CZ/EN.
- Produkční routování API přes Cloudflare Worker proxy je součástí nasazení frontendu.

### Architektura

```
┌─────────────────────────────────────────────────┐
│                   Frontend                        │
│         React + Vite + TailwindCSS               │
│    (SPA s 5-krokovým wizard průvodcem)            │
├─────────────────────────────────────────────────┤
│      Vite Dev Proxy / Cloudflare Worker Proxy     │
│ /discover,/extract,/train,/predict,… → backend    │
├─────────────────────────────────────────────────┤
│                   Backend                         │
│          Flask + Gunicorn (1 worker)              │
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
| Frontend | React, TypeScript, Vite, TailwindCSS | React 18, Vite 5 |
| Backend | Python, Flask, Gunicorn | Python 3.10, Flask 3.x |
| LLM | Ollama (lokální) — Qwen 2.5 VL 7B | ollama latest |
| Whisper | faster-whisper (large-v3, GPU, float16) | - |
| ML modely | RuleKit (Java), XGBoost, scikit-learn | xgboost 3.2 |
| Video | OpenCV, ffmpeg | cv2 4.x |
| Process manager | PM2 | - |
| Tunnel | Cloudflare (trycloudflare.com) | - |

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
│   └── ml_training.py          # Fáze 3/5: Ensemble training + prediction
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
│   ├── openai_service.py       # LLM API client (Ollama OpenAI-compatible)
│   └── speech_service.py       # Whisper transcription (faster-whisper)
└── utils/
    ├── file_utils.py           # ZIP extraction, media file discovery
    └── csv_utils.py            # CSV/labels loading utilities
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
| `model` | `RuleRegressor \| None` | Natrénovaný RuleKit model (jen regrese) |
| `xgb_model` | `XGBRegressor \| XGBClassifier \| None` | Natrénovaný XGBoost model |
| `scaler` | `StandardScaler \| None` | Feature scaler (in-memory) |
| `rules` | `list[str]` | Extrahovaná pravidla z RuleKit |
| `mse` | `float \| None` | Ensemble MSE na trénovacích datech |
| `rulekit_mse` | `float \| None` | RuleKit MSE |
| `xgb_mse` | `float \| None` | XGBoost MSE |
| `is_trained` | `bool` | Indikátor natrénovaného modelu |
| `testing_X` | `DataFrame \| None` | Extrahované features z testovacích dat |
| `_training_columns` | `list[str]` | Sloupce po preprocessing (pro alignment) |
| `_scaler_mean` | `list[float]` | Uložené parametry StandardScaler |
| `_scaler_scale` | `list[float]` | Uložené parametry StandardScaler |

#### Persistence

Pipeline stav se ukládá do `checkpoints/sessions/{session_id}/`:

| Soubor | Formát | Obsah |
|---|---|---|
| `pipeline_state.json` | JSON | Skalární metadata (feature_spec, rules, MSE, scaler params...) |
| `training_X.csv` | CSV | Extrahované trénovací features |
| `training_Y_df.csv` | CSV | Trénovací labels |
| `testing_X.csv` | CSV | Extrahované testovací features |
| `model.pkl` | Pickle | RuleKit Java model object |
| `xgb_model.pkl` | Pickle | XGBoost model |

**Migrační logika:** Pokud existuje legacy `pipeline_state.pkl` (starý pickle formát), systém ho automaticky načte, re-uloží v novém JSON+CSV formátu a smaže starý soubor.

---

### 2.3 Fáze 1: Feature Discovery

**Soubor:** `backend/pipeline/feature_discovery.py`

Dvouúrovňový proces pro automatický návrh feature specifikace:

#### Krok 1: Pozorování vzorků

Pro každý z max. 5 vzorkových médií se zavolá LLM s observation promptem:

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

Konfigurovatelné přes env var `EXTRACTION_PASSES` (default: 2).

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
- Server musí mít kompatibilní Python závislosti (načítání `model.pkl` / `xgb_model.pkl`).
- Pokud se interní formát checkpointu mezi verzemi změní, import může selhat.

Loguje statistiky: kolik hodnot bylo clampnuto, pro které features.

#### Imputace chybějících hodnot

Po extrakci všech řádků se chybějící hodnoty (`None`) nahradí **mediánem sloupce** (pro numerické). Toto je lepší než defaultní 0, protože:
- 0 může být validní hodnota na okraji škály
- Medián neposouvá distribuci

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

`StandardScaler` z scikit-learn:
- Fit na trénovacích datech: `scaler.fit_transform(X)`
- Transform na testovacích datech: `scaler.transform(X_test)`
- Parametry (`mean_`, `scale_`) se ukládají do `pipeline_state.json` pro rekonstrukci

#### Regrese: Ensemble RuleKit + XGBoost

Systém trénuje dva modely paralelně:

**RuleKit RuleRegressor:**
- Java-based rule induction (knihovna RuleKit)
- Pracuje na neskálovaných datech (rules jsou interpretabilní)
- Produkuje lidsky čitelná pravidla typu `IF feature > threshold THEN prediction`
- Výhoda: interpretabilita — uživatel vidí proč model predikuje danou hodnotu

**XGBoost Regressor:**
- Gradient boosted trees
- `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`
- Pracuje na škálovaných datech (StandardScaler)
- Výhoda: vysoká přesnost na tabulárních datech

**Ensemble predikce:**
```
ensemble = 0.4 × RuleKit + 0.6 × XGBoost
```

Váhy reflektují vyšší přesnost XGBoost při zachování interpretability z RuleKit.

#### Regrese: K-Fold Cross-Validation

Po natrénování na celých datech se spustí K-fold CV (k = min(5, počet vzorků)):

1. Pro každý fold: fit oba modely na train split, predict na val split
2. Compute ensemble MSE a MAE pro každý fold
3. Report: `cv_mse` (průměr), `cv_std` (rozptyl), `cv_mae`

**Overfitting detection:** Pokud `cv_mse > 2 × training_mse`, systém vrátí warning.

#### Feature Importance

Dvě perspektivy:
- **XGBoost:** `feature_importances_` (gain-based importance)
- **RuleKit:** Frekvence features v pravidlech (kolikrát se feature objeví v rule conditions)

Vrací se v API response pro vizualizaci ve frontend.

#### Klasifikace: validace cíle a model

Při `target_mode = classification` backend nejdřív validuje, že zvolený target sloupec opravdu vypadá jako kategorická proměnná:
- musí obsahovat alespoň 2 různé třídy
- nesmí být prázdný po odfiltrování chybějících hodnot
- pokud je téměř celý numerický a má vysokou kardinalitu, je považován za pravděpodobně spojitou proměnnou a backend vrátí chybu s doporučením přepnout na regresi
- pokud má vysokou kardinalitu i jako string a většina tříd je singleton nebo téměř singleton, je považován za identifier-like sloupec a backend ho odmítne

Model pro klasifikaci:
- `RuleClassifier` z RuleKitu
- vrací interpretovatelná pravidla i v klasifikačním režimu
- pokud je k dispozici `predict_proba`, frontend zobrazuje i confidence
- `StratifiedKFold` místo obyčejného `KFold`, aby foldy zachovávaly rozložení tříd

Metriky klasifikace:
- train: `accuracy`, `balanced_accuracy`, `f1_macro`, `mcc`
- cross-validation: `cv_accuracy`, `cv_balanced_accuracy`, `cv_f1_macro`, `cv_precision_macro`, `cv_recall_macro`, `cv_mcc`

V klasifikační větvi se nepoužívá `MSE`, `MAE` ani regresní ensemble s XGBoost.

#### Async progress fáze (aktuální implementace)

`train_model(..., progress_cb=...)` publikuje průběh po hlavních krocích:

- 10%: příprava tréninku
- 15%: normalizace features (`StandardScaler`)
- 25%: trénink RuleKit
- 60%: trénink XGBoost
- 75%: ensemble predikce
- 80%: K-fold cross-validation
- 98%: ukládání modelu/checkpointu
- 100%: dokončeno

#### Výstup tréninku (v `status.details` po dokončení jobu)

```json
{
  "status": "success",
  "mse": 0.0234,                    // Ensemble MSE (training)
  "rulekit_mse": 0.0312,            // RuleKit MSE
  "xgb_mse": 0.0189,               // XGBoost MSE
  "cv_mse": 0.0456,                // Cross-validated MSE
  "cv_std": 0.0089,                // CV standard deviation
  "cv_mae": 0.1523,                // CV MAE
  "cv_folds": 5,                   // Počet foldů
  "rules_count": 12,
  "rules": ["IF action_intensity > 5.5 AND speech_presence = 1 THEN 0.72", ...],
  "feature_importance": {
    "xgboost": {"action_intensity": 0.34, "visual_complexity": 0.28, ...},
    "rulekit": {"action_intensity": 0.42, "speech_presence": 0.33, ...}
  },
  "warnings": [],                   // Overfitting warnings
  "training_data_X": [...]          // Extrahovaná data pro UI
}
```

---

### 2.7 Fáze 5: Prediction

**Soubor:** `backend/pipeline/ml_training.py` → `predict_batch()`

**Route:** `backend/routes/predict.py` (`POST /predict`) běží asynchronně a vrací `job_id`.

1. Preprocessing testovacích features (alignment na training columns)
2. RuleKit predikce (neskálovaná data)
3. XGBoost predikce (škálovaná data přes uložený scaler)
4. Ensemble: `0.4 × RuleKit + 0.6 × XGBoost`
5. Pokud jsou k dispozici actual labels: MSE, MAE, Pearson correlation

Fallback: pokud `xgb_model` neexistuje (legacy modely), použije se jen RuleKit.

#### Klasifikační predikce

Pokud je pipeline v režimu klasifikace:
1. provede se preprocessing features stejně jako při tréninku
2. použije se `RuleClassifier`
3. vrací se `predicted_label`, volitelně `confidence` a také použité pravidlo, pokud se podaří dohledat pokrývající rule
4. pokud jsou k dispozici ground-truth labels, musí CSV obsahovat stejný target sloupec jako při tréninku; jinak backend vrátí chybu místo tichého fallbacku na jiný sloupec
5. pokud jsou k dispozici ground-truth labels, počítají se klasifikační metriky:
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
- 40%: XGBoost predikce
- 55%: ensemble kombinace
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

Session registry eviktuje neaktivní session po 6 hodinách (TTL-based cleanup).

#### Ollama EOF — popis chyby a oprava

**Chybová zpráva:**
```
Error code: 500 - {'error': {'message': 'do load request: Post "http://127.0.0.1:40125/load": EOF'}}
```

**Příčina:** Ollama používá interní HTTP server pro správu modelů. Při prvním requestu je model načten (`/load`). Pokud ve stejnou chvíli přijde druhý request (jiný uživatel nebo dvě otevřená okna), oba se snaží model načíst souběžně. Vnitřní Ollama server vrátí druhému requestu `EOF` — spojení bylo uzavřeno protože model byl již v procesu načítání.

Konkrétně k chybě dochází, pokud:
1. Uživatel A spustí Fázi 1 (Feature Discovery) → background thread čeká na Ollama
2. Uživatel B (nebo druhá záložka) spustí jakoukoliv fázi → další Ollama request

**Oprava:** Globální `threading.Semaphore(1)` v `backend/services/openai_service.py` serializuje všechna volání Ollama API. Druhý thread čeká, dokud první nedokončí svůj LLM call.

```python
_ollama_lock = threading.Semaphore(1)

# Každý LLM call je chráněn:
with _ollama_lock:
    response = local_client.chat.completions.create(...)
```

**Dopad na výkon:** Extrakce více médií jedním uživatelem probíhá sekvenčně (bylo tak i dříve). Souběžní uživatelé se řadí do fronty na úrovni LLM callů. Celková doba zpracování je stejná, ale bez crashů.

---

## 3. Frontend — Detailní popis

### 3.1 Architektura

React SPA s TypeScript, build systém Vite, styling TailwindCSS.

```
frontend/src/
├── App.tsx                         # Hlavní komponenta, routing train/predict mode
├── hooks/
│   ├── useTrainingPipeline.ts      # Custom hook — veškerý pipeline stav + handlery
│   └── usePollProgress.ts          # Generic polling hook pro async joby
├── components/
│   ├── TrainingView.tsx            # 5-krokový wizard
│   ├── PredictView.tsx             # Prediction-only mode
│   └── ...                         # UI komponenty
└── lib/
    └── api.ts                      # API URL konstanty, typy, style mappings
```

### 3.2 Custom Hook — `useTrainingPipeline`

**Soubor:** `frontend/src/hooks/useTrainingPipeline.ts`

Zapouzdřuje veškerý stav a logiku 5-fázového pipeline:

- **15+ useState hooks** pro fáze 1-5 (busy flags, data, výsledky)
- **7 handler funkcí** (discover, extract training/testing, train, predict, cancel, reset)
- **Generic `_runExtract(config)`** — eliminuje duplikaci 4 téměř identických extraction handlerů
- **localStorage persistence** — ukládá jen metadata (trainingStep, targetVariable, featureSpec, modelProvider), nikoliv velká data
- **Phase 3 + 5 polling** — trénink i predikce používají `job_id` + `pollProgress(...)` (už nejsou synchronní)

#### Předvýběr cílového sloupce

Ve Fázi 3 se po načtení `datasetYColumns` předvybírá **poslední** sloupec z CSV (typicky score/label), nikoliv první ID sloupec.

### 3.3 Polling mechanismus

**Soubor:** `frontend/src/hooks/usePollProgress.ts`

```typescript
pollProgress(jobId, onStatus, abortSignal)
```

Polluje `GET /status/{jobId}` každých N ms, volá callback s aktuálním stavem. Podporuje abort (zrušení uživatelem).

Při síťové chybě používá exponenciální backoff (od cca 600 ms do 5 s) a po úspěšném pollu delay resetuje.

### 3.4 API komunikace a session identifikace

Vite dev server proxyuje API requesty na backend:
```
/discover → http://127.0.0.1:5000/discover
/extract  → http://127.0.0.1:5000/extract
...
```

Podpora Cloudflare tunnelu: HMR disabled (`hmr: false`), timeout 100s v tunnel.

#### Cloudflare Worker proxy (produkční nasazení)

Pro doménu `*.workers.dev` je použit Worker (`frontend/worker.js`), který:

- proxyuje API cesty (`/discover`, `/extract`, `/train`, `/predict`, `/status`, `/state`, `/health`, `/queue-info`, ...)
- servíruje statická frontend aktiva přes `ASSETS`
- řeší CORS preflight (`OPTIONS`) a nastavuje CORS hlavičky na proxy odpovědích
- bufferuje request/response body (`arrayBuffer`) kvůli stabilnímu přenosu JSON payloadů (`job_id`) přes Worker runtime

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
3. Features se použijí pro klasický ML model (RuleKit/XGBoost)

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

### 4.3 Proč ensemble RuleKit + XGBoost?

| Vlastnost | RuleKit | XGBoost |
|---|---|---|
| Interpretabilita | Vysoká (if-then pravidla) | Nízká (black box) |
| Přesnost | Střední | Vysoká |
| Robustnost | Nižší (single model) | Vyšší (100 trees) |

Ensemble kombinuje výhody obou: RuleKit pravidla se zobrazují uživateli pro porozumění, XGBoost zvyšuje přesnost predikce. Váhy (0.4/0.6) reflektují tento kompromis.

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
| `VITE_API_BASE` | `""` | Frontend API base URL (prázdné = proxy) |
| `BACKEND_URL` | - | Backend URL pro Cloudflare Worker proxy (`frontend/wrangler.toml` → `[vars]`) |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,https://bpfeaturelab.kovm23.workers.dev` | CORS allowlist backendu pro browser requesty a Worker frontend |
| `FLASK_DEBUG` | `0` | Debug mode (0 = production) |
| `EXTRACTION_PASSES` | `2` | Počet LLM callů per médium při extrakci |

Poznámka: při použití `trycloudflare.com` je `BACKEND_URL` dočasná adresa, která se mění po restartu tunelu.

Poznámka: při produkčním nasazení přes `*.workers.dev` musí backend `ALLOWED_ORIGINS` obsahovat veřejný frontend origin, jinak polling na `/status/<job_id>` a `/queue-info` skončí 403.

### 6.2 PM2 procesy

| Proces | Příkaz | Port |
|---|---|---|
| `backend` | `gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 app:app` | 5000 |
| `frontend` | `npx vite --host --port 5173` | 5173 |
| `tunnel-backend` | Cloudflare tunnel → :5000 | — |
| `tunnel-frontend` | Cloudflare tunnel → :5173 | — |

### 6.3 Spuštění

```bash
# Backend
cd backend
source ../venv/bin/activate
pm2 start ../venv/bin/gunicorn --name backend -- -w 1 -b 0.0.0.0:5000 --timeout 600 app:app

# Frontend
cd frontend
pm2 start npx --name frontend -- vite --host --port 5173
```

---

## 7. API Reference

### POST /discover
Spustí feature discovery z vzorkových médií.

**Request:** `multipart/form-data`
- `files`: Media soubory (max 5)
- `target_variable`: Název cílové proměnné
- `model`: ID LLM modelu
- `labels_file` (optional): CSV s labels

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

### POST /predict
Spustí batch predikci.

**Request:** `multipart/form-data` (optional `labels_file`) nebo prázdný POST.

**Response (okamžitě):**
```json
{"job_id": "uuid"}
```

Finální výsledek je dostupný přes `GET /status/{job_id}` v `details`:
```json
{
  "predictions": [
    {
      "media_name": "video_001",
      "predicted_score": 0.672,
      "actual_score": 0.71,
      "rule_applied": "IF action_intensity > 5.5 THEN 0.672",
      "extracted_features": {...}
    }
  ],
  "metrics": {
    "mse": 0.0234,
    "mae": 0.112,
    "correlation": 0.89,
    "matched_count": 50,
    "total_count": 50
  }
}
```

### POST /reset
Vymaže všechny checkpointy a resetuje pipeline stav.

**Response:** `{"ok": true, "removed_checkpoints": [...]}`
