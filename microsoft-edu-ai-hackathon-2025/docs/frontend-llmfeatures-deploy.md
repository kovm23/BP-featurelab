# Frontend Deploy Na `llmfeatures.vse.cz`

Tento scénář počítá s tím, že:

- backend běží na jiném serveru
- `llmfeatures.vse.cz` hostuje pouze statický frontend build
- backend není běžně dostupný z internetu
- Apache servíruje `frontend/dist` a zároveň proxyuje `/api/*` na interní backend

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
VITE_API_BASE=/api
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
sudo a2enmod rewrite proxy proxy_http
sudo a2ensite llmfeatures.vse.cz.conf
sudo apache2ctl configtest
sudo systemctl reload apache2
```

Apache pak bude:

- servírovat SPA z `/var/www/llmfeatures/current`
- přeposílat `/api/*` na `http://llm.vse.cz:5000/*`

Tím pádem browser už nebude volat backend napřímo a nepotřebuje veřejnou backend URL.

## 4. Backend server `llm.vse.cz`

Na backend serveru musí běžet aplikace na portu `5000` a frontend server na ni musí mít síťový přístup.

V `backend/.env` doporučuji ponechat:

```bash
ALLOWED_ORIGINS=http://llmfeatures.vse.cz,https://llmfeatures.vse.cz
```

V proxy režimu to není hlavní integrační mechanismus, protože browser komunikuje se stejným originem `llmfeatures.vse.cz`, ale tato allowlist hodnota je pořád bezpečná a použitelná.

## 5. Co dělat s chybou `Temporary failure resolving 'proxy.vse.cz'`

Tato chyba nesouvisí s frontend buildem samotným, ale s DNS/proxy konfigurací serveru při `apt install`.

Důležité pro split deploy:

- frontend server kvůli statickému webu nepotřebuje backendové balíčky
- když build připravíš jinde, na `llmfeatures.vse.cz` nepotřebuješ ani `nodejs` a `npm`
- problém s `python3.13-venv` tedy neblokuje vytvoření prázdného Apache virtualhostu

## 6. Výsledek

Po těchto krocích bude:

- `http://llmfeatures.vse.cz/` servírovat frontend SPA
- frontend posílat API requesty na `http://llmfeatures.vse.cz/api/*`
- Apache na `llmfeatures.vse.cz` tyto requesty přepošle na interní backend `http://llm.vse.cz:5000/*`
- backend a frontend nasazené odděleně bez nutnosti zveřejnit backend přímo do internetu
