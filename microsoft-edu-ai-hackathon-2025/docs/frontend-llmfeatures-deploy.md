# Frontend Deploy Na `llmfeatures.vse.cz`

Tento scénář počítá s tím, že:

- backend běží na jiném serveru
- `llmfeatures.vse.cz` hostuje pouze statický frontend build
- backend není běžně dostupný z internetu
- Apache servíruje `frontend/dist` a zároveň proxyuje `/api/*` na interní backend
- veřejné HTTPS je ukončené na `llmfeatures.vse.cz`

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

Konfigurace je připravena v [apache-llmfeatures.vse.cz.conf](apache-llmfeatures.vse.cz.conf).

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

## 4. Zapni HTTPS přes Certbot

Po ověření, že web funguje přes HTTP, nainstaluj Certbot:

```bash
sudo apt update
sudo apt install -y certbot python3-certbot-apache
```

V tomto prostředí je v DNS dostupná doména:

- `llmfeatures.vse.cz`

Naopak `www.llmfeatures.vse.cz` zde DNS záznam nemá, takže certifikát vystavuj jen pro hlavní doménu:

```bash
sudo certbot --apache -d llmfeatures.vse.cz
```

Po úspěšném vydání certifikátu Certbot:

- nasadí certifikát do Apache
- přidá HTTP -> HTTPS redirect
- nastaví automatickou obnovu přes `certbot.timer`

Poznámka:

- Certbot v tomto scénáři typicky zapisuje SSL konfiguraci do `/etc/apache2/sites-available/000-default-ssl.conf`
- po jeho zásahu musí v HTTPS virtualhostu zůstat i reverse proxy pro `/api/*`

Ověření:

```bash
sudo apache2ctl configtest
curl -I https://llmfeatures.vse.cz
curl -I https://llmfeatures.vse.cz/api/health
```

## 5. Backend server `llm.vse.cz`

Na backend serveru musí běžet aplikace na portu `5000` a frontend server na ni musí mít síťový přístup.

V `backend/.env` doporučuji ponechat:

```bash
ALLOWED_ORIGINS=http://llmfeatures.vse.cz,https://llmfeatures.vse.cz
```

V proxy režimu to není hlavní integrační mechanismus, protože browser komunikuje se stejným originem `llmfeatures.vse.cz`, ale tato allowlist hodnota je pořád bezpečná a použitelná.

## 6. Co dělat s chybou `Temporary failure resolving 'proxy.vse.cz'`

Tato chyba nesouvisí s frontend buildem samotným, ale s DNS/proxy konfigurací serveru při `apt install`.

Důležité pro split deploy:

- frontend server kvůli statickému webu nepotřebuje backendové balíčky
- když build připravíš jinde, na `llmfeatures.vse.cz` nepotřebuješ ani `nodejs` a `npm`
- problém s `python3.13-venv` tedy neblokuje vytvoření prázdného Apache virtualhostu

Poznámka z reálného nasazení:

- na školní síti nepoužívej externí nameservery typu `8.8.8.8`
- resolver musí zůstat na školních DNS serverech

## 7. Výsledek

Po těchto krocích bude:

- `https://llmfeatures.vse.cz/` servírovat frontend SPA
- frontend posílat API requesty na `https://llmfeatures.vse.cz/api/*`
- Apache na `llmfeatures.vse.cz` tyto requesty přepošle na interní backend `http://llm.vse.cz:5000/*`
- backend a frontend nasazené odděleně bez nutnosti zveřejnit backend přímo do internetu
