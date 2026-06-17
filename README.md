# 🛡️ SpamShield — SMS Spam Classifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sms-spam-classifier-qpnp2zesr287tny2mmidfa.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** with real-time confidence scores. Built with a Scikit-Learn model backend served via a FastAPI REST API and a polished Streamlit frontend.

---

## 🔗 Live Demo

👉 **[Try it here](https://sms-spam-classifier-qpnp2zesr287tny2mmidfa.streamlit.app/)**

Paste any SMS or message and get an instant verdict with spam/ham probability scores.

---

## ✨ Features

- **Real-time classification** — Instantly detects Spam or Ham with a single click
- **Confidence scores** — Returns both spam and ham probabilities with visual progress bars
- **NLP preprocessing pipeline** — Lowercasing, tokenization, stopword removal, and Porter stemming
- **FastAPI backend** — Clean REST API with Pydantic-validated request/response schemas
- **MLflow model tracking** — Experiments tracked and models logged with MLflow
- **Streamlit frontend** — A polished, dark-themed UI with animated verdict banners
- **Decoupled architecture** — Frontend and backend are independently deployable

---

## 🏗️ Project Structure

```
sms-spam-classifier/
│
├── app.py                  # FastAPI backend (REST API)
├── frontend.py             # Streamlit frontend UI
├── main.py                 # Entry point / pipeline runner
├── setup.py                # Package setup
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── schemas/
│   │   └── prediction_schema.py   # Pydantic request/response models
│   ├── services/
│   │   └── prediction_service.py  # Model loading & inference logic
│   └── exception.py               # Custom exception handling
│
├── model/                  # Serialized model and vectorizer artifacts
├── artifacts/              # Pipeline output artifacts
├── notebook/               # EDA and model training Jupyter notebooks
│
├── .gitignore
└── LICENSE
```

---

## 🧠 How It Works

### NLP Preprocessing Pipeline

Each incoming message is processed through a multi-step pipeline before inference:

1. **Lowercasing** — Normalizes text to lowercase
2. **Tokenization** — Splits text into individual tokens using NLTK
3. **Alphanumeric filtering** — Removes punctuation and special characters
4. **Stopword removal** — Drops common English stopwords (e.g. "the", "is")
5. **Porter Stemming** — Reduces words to their root form (e.g. "winning" → "win")

### Model

The classifier is trained using Scikit-Learn on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) and tracked with **MLflow**. The trained model and TF-IDF vectorizer are serialized and loaded at inference time.

### API

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|--------|------------------|------------------------------|
| `GET` | `/` | Health check / home |
| `GET` | `/health` | Returns model status & version |
| `POST` | `/api/v1/predict` | Classifies a message |

**Request body (`/api/v1/predict`):**
```json
{
  "text": "Congratulations! You've won a FREE prize. Call now!"
}
```

**Response:**
```json
{
  "text": "Congratulations! You've won a FREE prize. Call now!",
  "prediction": "Spam",
  "spam_probability": 0.97,
  "ham_probability": 0.03
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Dakshhhhh-ops/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 4. Run the FastAPI backend

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

### 5. Run the Streamlit frontend

```bash
streamlit run frontend.py
```

> **Note:** The frontend is configured to call the hosted API at `https://sms-spam-classifier-2-t2cs.onrender.com/api/v1/predict`. To point it at your local backend, update the `API_URL` variable in `frontend.py`.

---

## 📦 Dependencies

| Package | Purpose |
|-------------|----------------------------------|
| `scikit-learn` | ML model training & inference |
| `nltk` | Text preprocessing |
| `fastapi` | REST API framework |
| `pydantic` | Schema validation |
| `mlflow` | Experiment tracking |
| `streamlit` | Frontend UI |
| `xgboost` | Gradient boosting classifier |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | EDA visualizations |
| `requests` | HTTP calls from the frontend |

---

## 🧪 Running the Training Pipeline

To retrain the model from scratch, run the pipeline via:

```bash
python main.py
```

Training notebooks with EDA and model experiments are available in the `notebook/` directory.

---

## 📂 Key Files

| File | Description |
|---|---|
| `app.py` | FastAPI app with `/predict` endpoint |
| `frontend.py` | Streamlit UI with dark theme, verdict banners, and confidence bars |
| `main.py` | Pipeline orchestration script |
| `src/services/prediction_service.py` | Loads model and runs inference |
| `src/schemas/prediction_schema.py` | Pydantic models for API I/O |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Daksh**  
GitHub: [@Dakshhhhh-ops](https://github.com/Dakshhhhh-ops)
