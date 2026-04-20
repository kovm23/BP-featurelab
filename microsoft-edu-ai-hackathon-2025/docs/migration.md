# Migrace Na Jiný Server

Doporučený scénář pro produkci: frontend na `llmfeatures.vse.cz` (Apache), backend + Ollama na serveru `llm.vse.cz`.

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

Aktualizuj `ALLOWED_ORIGINS` v `backend/.env` na novou frontend doménu. Pak znovu načti konfiguraci:

```bash
pm2 reload ecosystem.config.js
```

## 8. Split Deploy: Backend Jinde, Frontend Na `llmfeatures.vse.cz`

Pokud backend poběží na jednom serveru a `llmfeatures.vse.cz` bude hostovat jen webovou aplikaci, odděl to takto:

- backend server: Python, Java, `ffmpeg`, Ollama, PM2
- frontend server: Apache, statický build z `frontend/dist` a reverse proxy `/api/*` na interní backend

Pro frontend-only server použij [frontend-llmfeatures-deploy.md](frontend-llmfeatures-deploy.md) a Apache konfiguraci [apache-llmfeatures.vse.cz.conf](apache-llmfeatures.vse.cz.conf).

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
6. Aktualizuj `ALLOWED_ORIGINS` na novou frontend doménu a `pm2 reload ecosystem.config.js`.
