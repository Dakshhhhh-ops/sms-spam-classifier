# 🛡️ SpamShield — SMS Spam Classifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://sms-spam-classifier-qpnp2zesr287tny2mmidfa.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)](https://scikit-learn.org)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge\&logo=render\&logoColor=black)](https://render.com)

# 📌 Overview

SpamShield is an end-to-end machine learning application that classifies SMS messages as **Spam** or **Ham (Not Spam)** in real time.

The project combines:

* NLP preprocessing with NLTK
* TF-IDF feature extraction
* Scikit-Learn classification model
* FastAPI REST API backend
* Streamlit frontend
* Cloud deployment using Render and Streamlit Cloud

Users can submit any SMS message and instantly receive a prediction along with confidence scores.

---

# 🔗 Live Demo

### Frontend

https://sms-spam-classifier-qpnp2zesr287tny2mmidfa.streamlit.app/

### Backend API

https://sms-spam-classifier-2-t2cs.onrender.com

### Swagger Documentation

https://sms-spam-classifier-2-t2cs.onrender.com/docs

---

# ✨ Features

* Real-time SMS spam detection
* Spam and Ham confidence scores
* NLP preprocessing pipeline
* FastAPI-powered REST API
* Streamlit interactive frontend
* Pydantic schema validation
* Modular service-based architecture
* Cloud deployment

---

# 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI Backend
 │
 ▼
Prediction Service
 │
 ▼
Predict Pipeline
 │
 ▼
TF-IDF Vectorizer
 │
 ▼
Scikit-Learn Model
 │
 ▼
Prediction Response
```

---

# 📂 Project Structure

```text
sms-spam-classifier/
│
├── app.py
├── frontend.py
├── requirements.txt
├── setup.py
│
├── artifacts/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── exception.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── schemas/
│   │   └── prediction_schema.py
│   │
│   ├── services/
│   │   └── prediction_service.py
│   │
│   └── utils.py
│
├── notebook/
│
└── README.md
```

---

# 🧠 NLP Pipeline

Every incoming message passes through the following preprocessing pipeline:

### 1. Lowercasing

Converts text to lowercase for consistency.

### 2. Tokenization

Splits the message into individual words using NLTK.

### 3. Alphanumeric Filtering

Removes punctuation and special characters.

### 4. Stopword Removal

Removes common English stopwords.

Example:

```text
the
is
a
and
```

### 5. Porter Stemming

Converts words to their root forms.

Example:

```text
winning → win
running → run
played → play
```

### 6. TF-IDF Vectorization

Transforms processed text into numerical features suitable for machine learning.

---

# 🤖 Model

The model is trained using the UCI SMS Spam Collection dataset.

The trained classifier and vectorizer are serialized and stored as:

```text
artifacts/model.pkl
artifacts/vectorizer.pkl
```

These artifacts are loaded during application startup and used for real-time inference.

---

# 🚀 API Endpoints

| Method | Endpoint        | Description          |
| ------ | --------------- | -------------------- |
| GET    | /               | Home endpoint        |
| GET    | /health         | Service health check |
| POST   | /api/v1/predict | Predict spam or ham  |

---

## Request

```json
{
  "text": "Congratulations! You've won a FREE prize."
}
```

---

## Response

```json
{
  "text": "Congratulations! You've won a FREE prize.",
  "prediction": "Spam",
  "spam_probability": 0.97,
  "ham_probability": 0.03
}
```

---

# 🚀 Local Setup

## Clone Repository

```bash
git clone https://github.com/Dakshhhhh-ops/sms-spam-classifier.git
cd sms-spam-classifier
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Download NLTK Resources

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Start FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

## Start Streamlit Frontend

```bash
streamlit run frontend.py
```

---

# 📦 Tech Stack

### Machine Learning

* Scikit-Learn
* NLTK
* NumPy
* Pandas

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Deployment

* Render
* Streamlit Cloud

---

# 📸 Screenshots

Add screenshots of:

* Spam prediction
* Ham prediction
* Swagger API documentation

These significantly improve recruiter impressions.

---

# 🎯 Key Highlights

* Built a complete end-to-end ML application
* Exposed model inference through REST APIs
* Designed a modular prediction pipeline
* Implemented schema validation using Pydantic
* Deployed backend and frontend independently
* Added confidence-based predictions with visual feedback

---

# 👤 Author

Daksh Wadhwa

GitHub:
https://github.com/Dakshhhhh-ops

LinkedIn:
(Add your LinkedIn profile link here)
