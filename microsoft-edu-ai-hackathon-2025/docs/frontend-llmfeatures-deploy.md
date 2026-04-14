# Frontend Deploy Na `llmfeatures.vse.cz`

Tento scénář počítá s tím, že:

- backend běží na jiném serveru
- `llmfeatures.vse.cz` hostuje pouze statický frontend build
- Apache zde neslouží jako reverse proxy na Flask, jen servíruje `frontend/dist`

## Co na frontend serveru opravdu potřebuješ

Minimum:

- `apache2`
- build frontendu (`frontend/dist`)

Volitelné:

- `nodejs` + `npm`, pokud chceš frontend buildovat přímo na tomto serveru

Na frontend serveru **nepotřebuješ**:

- Python venv
- Java
- `ffmpeg`
- Ollama
- backend `.env`

## 1. Připrav frontend build

Pokud build děláš přímo na cílovém serveru:

```bash
cd ~/BP/microsoft-edu-ai-hackathon-2025/frontend
cp .env.example .env.production.local
```

Do `frontend/.env.production.local` nastav:

```bash
VITE_API_BASE=https://BACKEND_PUBLIC_ORIGIN
```

Příklad:

```bash
VITE_API_BASE=https://llm-backend.vse.cz
```

Pak frontend buildni:

```bash
npm install
npm run build
```

Pokud build děláš jinde, stačí na `llmfeatures.vse.cz` nahrát hotový obsah složky `frontend/dist/`.

## 2. Připrav webroot

```bash
sudo mkdir -p /var/www/llmfeatures/current
sudo rsync -av --delete ~/BP/microsoft-edu-ai-hackathon-2025/frontend/dist/ /var/www/llmfeatures/current/
sudo chown -R www-data:www-data /var/www/llmfeatures
```

## 3. Aktivuj Apache virtualhost

Konfigurace je připravena v [apache-llmfeatures.vse.cz.conf](/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/docs/apache-llmfeatures.vse.cz.conf).

```bash
sudo cp ~/BP/microsoft-edu-ai-hackathon-2025/docs/apache-llmfeatures.vse.cz.conf /etc/apache2/sites-available/llmfeatures.vse.cz.conf
sudo a2enmod rewrite
sudo a2ensite llmfeatures.vse.cz.conf
sudo apache2ctl configtest
sudo systemctl reload apache2
```

## 4. CORS na backend serveru

Protože frontend poběží na jiné doméně, backend musí povolit:

```bash
ALLOWED_ORIGINS=http://llmfeatures.vse.cz,https://llmfeatures.vse.cz
```

Pokud backend poběží jen přes HTTP, ponech jen HTTP variantu. Pokud bude přidané HTTPS, nech obě.

## 5. Co dělat s chybou `Temporary failure resolving 'proxy.vse.cz'`

Tato chyba nesouvisí s frontend buildem samotným, ale s DNS/proxy konfigurací serveru při `apt install`.

Důležité pro split deploy:

- frontend server kvůli statickému webu nepotřebuje backendové balíčky
- když build připravíš jinde, na `llmfeatures.vse.cz` nepotřebuješ ani `nodejs` a `npm`
- problém s `python3.13-venv` tedy neblokuje vytvoření prázdného Apache virtualhostu

## 6. Výsledek

Po těchto krocích bude:

- `http://llmfeatures.vse.cz/` servírovat frontend SPA
- frontend posílat API requesty na backend server přes `VITE_API_BASE`
- backend a frontend nasazené odděleně
