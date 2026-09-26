# 🚀 PRODUCTION DEPLOYMENT GUIDE

This guide provides step-by-step instructions for deploying the **Natural Disaster Intelligence Dashboard** on **Streamlit Cloud** or **Heroku**.

---

## 🌐 Option 1: Deploying on Streamlit Cloud (Recommended - Free & Fast)

Streamlit Cloud provides seamless, native hosting for Streamlit applications directly from GitHub.

### Step 1: Push Repository to GitHub
Ensure your repository is pushed to GitHub:
```bash
git add .
git commit -m "Deploy production ready disaster dashboard"
git push origin main
```

### Step 2: Sign in to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Click **"New app"**.

### Step 3: App Configuration
- **Repository:** `Jyothika30-2006/data_visualization`
- **Branch:** `main` (or `arena/01a0b023-data-visualization`)
- **Main file path:** `app.py`

### Step 4: Configure Secrets & Environment Variables
In the Streamlit Cloud deployment settings, click **"Advanced settings..."** and paste your secrets in TOML format:

```toml
DB_PATH = "data/disaster_intelligence.db"
OPENWEATHER_API_KEY = "your_openweather_key"
TWILIO_ACCOUNT_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_token"
TWILIO_PHONE_NUMBER = "+1234567890"
SMTP_EMAIL = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
```

### Step 5: Deploy!
Click **"Deploy!"**. Streamlit Cloud will automatically install dependencies from `requirements.txt`, run `scripts/init_db.py` on first start if configured, and host your live app at `https://<your-app-name>.streamlit.app`.

---

## ☁️ Option 2: Deploying on Heroku / Container Hosting

### Step 1: Create `Procfile`
Create a file named `Procfile` in the project root:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Step 2: Set Heroku Config Vars
```bash
heroku config:set DB_PATH=data/disaster_intelligence.db
heroku config:set OPENWEATHER_API_KEY=your_key
```

### Step 3: Deploy to Heroku
```bash
git push heroku main
```

---

## ✅ Deployment Pre-Flight Testing Checklist

Before presenting or going live, verify:
- [x] Database seeded with `python scripts/init_db.py` (Verify `data/disaster_intelligence.db` exists).
- [x] All unit tests pass with `pytest tests/test_modules.py`.
- [x] Folium maps load smoothly without missing tile errors.
- [x] Live USGS API sync executes without error.
- [x] CSV export button downloads valid incident logs.
- [x] Nearest shelter search produces accurate Haversine distances.
