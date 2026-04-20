# Migrace Na Jiný Server

Nejjednodušší doporučený scénář:

- frontend ponechat na `workers.dev`
- přesouvat jen backend + Ollama
- po migraci přepnout pouze `BACKEND_URL` ve Workeru

## 1. Co musí být na novém serveru

- Python 3
- Node.js + PM2
- `ffmpeg`
- Java (kvůli RuleKitu)
- Ollama

## 2. Přenos projektu

Na nový server přenes:

- celý repozitář
- volitelně `backend/checkpoints/`, pokud chceš zachovat session a checkpointy

## 3. Bootstrap backendu

V rootu projektu spusť:

```bash
bash scripts/bootstrap_backend.sh
```

Script:

- vytvoří `venv`
- nainstaluje backend závislosti
- vytvoří `backend/.env` z `backend/.env.example`, pokud ještě neexistuje

## 4. Backend konfigurace

Zkontroluj `backend/.env`:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `ALLOWED_ORIGINS`
- volitelně `SECRET_KEY`

Poznámka:
- `OLLAMA_MODEL` je jen fallback výchozí model
- UI může stále poslat jiný model explicitně

## 5. Spuštění backendu

```bash
cd backend
pm2 start ecosystem.config.js
```

Backend poběží na portu `5000`.

## 6. Přepnutí frontendu

Pokud používáš Cloudflare Worker frontend:

1. nastav nový `BACKEND_URL`
2. redeployni Worker

Poznámka:
- `trycloudflare.com` URL je dočasná. Po restartu tunelu se změní a stará adresa přestane fungovat.
- Nekomituj do `frontend/wrangler.toml` starou tunelovou URL jako trvalou hodnotu. Před deployem ji vždy přepiš na aktuální backend origin.

Frontend UI není nutné přesouvat na nový server, pokud ti stačí stávající `workers.dev`.

## 8. Split Deploy: Backend Jinde, Frontend Na `llmfeatures.vse.cz`

Pokud backend poběží na jednom serveru a `llmfeatures.vse.cz` bude hostovat jen webovou aplikaci, odděl to takto:

- backend server: Python, Java, `ffmpeg`, Ollama, PM2
- frontend server: Apache, statický build z `frontend/dist` a reverse proxy `/api/*` na interní backend

Pro frontend-only server použij [frontend-llmfeatures-deploy.md](/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/docs/frontend-llmfeatures-deploy.md) a Apache konfiguraci [apache-llmfeatures.vse.cz.conf](/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/docs/apache-llmfeatures.vse.cz.conf).

Důležitá poznámka:

- na frontend serveru nepotřebuješ `python3.13-venv`, Java ani `ffmpeg`
- pokud frontend buildneš jinde a nahraješ jen `frontend/dist`, nepotřebuješ tam ani `nodejs` a `npm`
- frontend build má v tomto režimu používat `VITE_API_BASE=/api`
- Apache na `llmfeatures.vse.cz` pak přeposílá `/api/*` na interní backend, takže backend nemusí být veřejně dostupný z internetu
- veřejné HTTPS na `llmfeatures.vse.cz` můžeš zapnout přes `certbot --apache -d llmfeatures.vse.cz`
- pokud `www.llmfeatures.vse.cz` nemá DNS záznam, nevystavuj certifikát pro `www`
- backend může mít v `ALLOWED_ORIGINS` alespoň `http://llmfeatures.vse.cz` a případně i `https://llmfeatures.vse.cz`

## 7. Nejkratší migrační checklist

1. Přenes repo na nový server.
2. Nainstaluj `ffmpeg`, Java, Ollama, Python, Node.js, PM2.
3. Spusť `bash scripts/bootstrap_backend.sh`.
4. Uprav `backend/.env`.
5. Spusť `cd backend && pm2 start ecosystem.config.js`.
6. Přepni Worker `BACKEND_URL` na nový backend.
