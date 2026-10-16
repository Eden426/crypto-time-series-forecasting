# Deployment Guide

This guide walks through deploying the three parts of **Crypto Time-Series Forecasting**:

1. **Models & Scalers** → **Hugging Face Hub**
2. **FastAPI Backend** → **Render**
3. **Frontend UI** → **Vercel**

---

## Step 1: Upload Models to Hugging Face Hub

The trained GRU models (`*_gru.keras`) and scalers (`*_scaler.pkl`) in `models/` can be hosted on a Hugging Face Model repository and dynamically downloaded by the backend.

1. Create a **Write** access token on Hugging Face:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token with **Write** permissions.
2. Install `huggingface_hub` in your virtual environment (if not already installed):
   ```powershell
   pip install huggingface_hub
   ```
3. Run the upload script from the project root:
   ```powershell
   python scripts/upload_to_hf.py --repo-id <YOUR_HF_USERNAME>/crypto-time-series-gru --token <YOUR_HF_WRITE_TOKEN>
   ```
   This automatically creates the repository, uploads the Model Card (`models/MODEL_CARD.md` as `README.md`), and uploads all 4 `.keras` models and 4 `.pkl` scalers.

---

## Step 2: Deploy the FastAPI Backend to Render

The repository includes both a [`render.yaml`](./render.yaml) Blueprint and a [`Dockerfile`](./Dockerfile).

### Option A: Deploy via Render Blueprint (`render.yaml`)
1. Push your latest changes to GitHub.
2. Log in to [Render](https://dashboard.render.com/) → click **New +** → **Blueprint**.
3. Connect your GitHub repository (`Crypto-Time-Series-Forecasting`).
4. Render will detect `render.yaml` and prompt you for:
   - `HF_REPO_ID`: Set this to `<YOUR_HF_USERNAME>/crypto-time-series-gru` (from Step 1).
5. Click **Apply** to deploy.

### Option B: Deploy via Render Web Service (Manual)
1. In the Render dashboard, click **New +** → **Web Service** and connect your GitHub repository.
2. Configure the service:
   - **Runtime:** `Python 3` (or `Docker` to use the root `Dockerfile`)
   - **Build Command:** `pip install --upgrade pip && pip install -r backend/requirements.txt`
   - **Start Command:** `uvicorn backend.src.main:app --host 0.0.0.0 --port $PORT`
3. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.9`
   - `HF_REPO_ID` = `<YOUR_HF_USERNAME>/crypto-time-series-gru`
   - `ALLOWED_ORIGINS` = `*` (or your Vercel URL once deployed)
4. Once deployed, copy your Render service URL (e.g. `https://crypto-forecasting-backend.onrender.com`) and verify it by opening:
   - `https://<your-render-service>.onrender.com/health`

---

## Step 3: Deploy the Frontend to Vercel

1. Open [`frontend/src/index.js`](./frontend/src/index.js) and ensure `PRODUCTION_API_URL` at the top of the file matches your Render backend URL:
   ```js
   const PRODUCTION_API_URL = "https://crypto-forecasting-backend.onrender.com";
   ```
   *(If your Render service has a different subdomain, update this line, commit, and push to GitHub.)*
2. Log in to [Vercel](https://vercel.com/new) and import your GitHub repository.
3. You can deploy with **default settings** (root [`vercel.json`](./vercel.json) automatically routes `/`, `/style.css`, and `/index.js` to `frontend/src/`), **or** set the **Root Directory** to `frontend/src`.
4. Click **Deploy**. Your frontend is now live on Vercel and connected to your Render FastAPI backend and Hugging Face Hub models!
