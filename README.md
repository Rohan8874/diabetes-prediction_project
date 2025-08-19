

## Quick Links (replace after deploy)

- **Backend API (Render):** `https://YOUR-RENDER-URL.onrender.com`
- **Streamlit App:** `https://YOUR-STREAMLIT-URL`
- **Static Web Frontend (Netlify/Vercel/Pages):** `https://YOUR-FRONTEND-URL`

---

## Dataset

- **Pima Indians Diabetes Database (diabetes.csv):** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
  Download `diabetes.csv` and place it at: `data/diabetes.csv`.

---


---

## Requirements

- **Python 3.11** (recommended)  
- **pip** (latest)  
- **Docker Desktop** (for container & Render parity)
- (Optional) **Streamlit Community Cloud** or any static host for the frontend

Install Python dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

```

`requirements.txt` includes:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic>=2.6,<3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
python-multipart==0.0.9
requests==2.32.3
streamlit==1.36.0
anyio==4.4.0
```

---

## 1) Train the Model

The training script:
- Treats biologically-impossible zeros as missing (imputes median) for: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- Trains **5 models**: Logistic Regression, Random Forest, SVM, Decision Tree, KNN
- Selects **best by weighted F1** on the hold-out test set
- Saves:
  - `model/diabetes_model.pkl` (joblib bundle: pipeline + metadata)
  - `metrics/metrics.json` (all metrics + classification report)

Run:

```bash
python training/train.py
```

Expected output:
- Console prints best model and metrics
- Files created:
  - `model/diabetes_model.pkl`
  - `metrics/metrics.json`

---

## 2) Run the FastAPI Server (Locally)

Start FastAPI:

```bash
uvicorn api.main:app --reload
```

Open:
- Swagger Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Endpoints

**GET `/health`**
```json
{ "status": "ok" }
```

**POST `/predict`**  
Request:
```json
{
  "Pregnancies": 3,
  "Glucose": 145,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.35,
  "Age": 29
}
```
Response:
```json
{
  "prediction": 0,
  "result": "Not Diabetic",
  "confidence": 0.87
}
```

**GET `/metrics`**  
Returns the saved test metrics JSON.

### Example Requests

**PowerShell (recommended):**
```powershell
# Health
irm "http://127.0.0.1:8000/health"

# Predict
irm "http://127.0.0.1:8000/predict" -Method Post -ContentType 'application/json' -Body '{"Pregnancies":3,"Glucose":145,"BloodPressure":70,"SkinThickness":20,"Insulin":85,"BMI":33.6,"DiabetesPedigreeFunction":0.35,"Age":29}'
```

**curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{"Pregnancies":3,"Glucose":145,"BloodPressure":70,"SkinThickness":20,"Insulin":85,"BMI":33.6,"DiabetesPedigreeFunction":0.35,"Age":29}'
```

---

## 3) Docker (Build & Run)

> Ensure `model/diabetes_model.pkl` and `metrics/metrics.json` exist **before** building.

```bash
# Build
docker build -t diabetes-api:latest .

# Run
docker run --name diabetes-api -p 8000:8000 diabetes-api:latest

# Test
curl http://127.0.0.1:8000/health
```

**Optional: docker-compose (dev hot-reload)**
```bash
docker compose up --build
```

---

## 4) Deploy to Render

1. Push your repo to GitHub.
2. In **Render**: New → **Web Service** → Connect repo
3. Settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Deploy → Note your live URL, e.g. `https://your-service.onrender.com`
5. Verify:
   - `GET https://your-service.onrender.com/health`
   - `POST https://your-service.onrender.com/predict`

> CORS is open to `*` in this demo. For production, restrict to your frontend domain.

---

## 5) Frontend Options

### A) Streamlit (easiest)

Run locally:

```bash
# If API is running locally on 8000
# Option 1: use env var (no secrets file)
# Windows PowerShell:
$env:API_URL="http://127.0.0.1:8000"


streamlit run frontend/streamlit_app.py
```

**OR** create a secrets file (avoids env var):

Create `frontend/.streamlit/secrets.toml`:
```toml
API_URL = "http://127.0.0.1:8000"
```

Deploy on **Streamlit Community Cloud**:
- App path: `frontend/streamlit_app.py`
- Add a secret: `API_URL = "https://your-service.onrender.com"`

### B) Static Web (HTML + JS)

Serve locally:
```bash
cd frontend/web
python -m http.server 5500
# open http://127.0.0.1:5500
```

Set the backend URL in `frontend/web/script.js`:
```js
const API_URL = localStorage.getItem("API_URL") || "http://127.0.0.1:8000";
```
For production, set it in the browser console once:
```js
localStorage.setItem("API_URL","https://your-service.onrender.com")
```
Host on Netlify, Vercel, or GitHub Pages.

---

## 6) Evaluation Metrics

The training script prints and saves:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score (weighted, used for model selection)**
- Full `classification_report` per class

Check them at runtime:
- **API**: `GET /metrics`
- **Streamlit**: expand the “Metrics” panel

---

## 7) Notes & Design Choices

- **Preprocessing**: median imputation for zeros in certain biomedical fields; scaling where needed.
- **Model selection**: best weighted F1 on test set.
- **Confidence**: uses `predict_proba` if available; else a conservative fallback.
- **Feature order** is recorded in the model **metadata** and enforced during inference.

---

## 8) Troubleshooting (Windows-friendly)

**A. Docker named pipe errors / daemon not running**
- Start Docker Desktop and wait for “Docker Desktop is running”.
- Clear bad `DOCKER_HOST`:
  ```powershell
  [Environment]::SetEnvironmentVariable("DOCKER_HOST",$null,"User")
  [Environment]::SetEnvironmentVariable("DOCKER_HOST",$null,"Machine")
  ```
  Reopen PowerShell.
- Ensure WSL2 is installed/in use:
  ```powershell
  wsl --install
  wsl --update
  wsl --set-default-version 2
  wsl --shutdown
  ```
- Add your user to `docker-users` group:
  ```powershell
  net localgroup docker-users "$env:USERNAME" /add
  ```
  Sign out/in.

**B. `pull access denied for diabetes-api`**
- You didn’t build locally. Run:
  ```powershell
  docker build -t diabetes-api:latest .
  docker run -p 8000:8000 diabetes-api:latest
  ```

**C. PowerShell `Invoke-WebRequest` header error**
- Use `Invoke-RestMethod`:
  ```powershell
  irm "http://127.0.0.1:8000/predict" -Method Post -ContentType 'application/json' -Body '{"Pregnancies":3,"Glucose":145,"BloodPressure":70,"SkinThickness":20,"Insulin":85,"BMI":33.6,"DiabetesPedigreeFunction":0.35,"Age":29}'
  ```

**D. Streamlit `StreamlitSecretNotFoundError`**
- Either set env var `API_URL` before `streamlit run`, **or**
- Create `frontend/.streamlit/secrets.toml`:
  ```toml
  API_URL = "http://127.0.0.1:8000"
  ```

**E. Port already in use**
```powershell
# choose a different host port
docker run -p 9000:8000 diabetes-api:latest
# then open http://127.0.0.1:9000/health
```

**F. Model not found inside container**
- Make sure you trained and committed the files **before** `docker build`, **or** mount them:
  ```powershell
  docker run -p 8000:8000 -v "$PWD/model:/app/model" -v "$PWD/metrics:/app/metrics" diabetes-api:latest
  ```

---

## 9) Submission Checklist

- [ ] Trained model with ≥2 classifiers
- [ ] Metrics printed + `metrics/metrics.json` saved
- [ ] Best model saved: `model/diabetes_model.pkl`
- [ ] Async FastAPI app (`/health`, `/predict`, `/metrics`)
- [ ] Dockerfile builds & runs the API
- [ ] API deployed to Render (live link)
- [ ] Frontend (Streamlit or HTML/JS) working with API (live link)
- [ ] Clean GitHub repo with the structure above + this README


## Acknowledgments

- Dataset: UCI / Kaggle — Pima Indians Diabetes Database.
