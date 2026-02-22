# verifyNEWS — The Daily Verifier

A Streamlit-based fake news screening app that predicts whether a pasted news article is likely fake or not, using a pre-trained TF-IDF + Logistic Regression pipeline.

## Overview

`verifyNEWS` includes:

- An interactive Streamlit UI (`app.py`) for real-time prediction
- A training/experimentation notebook (`news.ipynb`)
- Pre-trained model artifacts (`model.joblib`, `vectorizer.joblib`)
- Example datasets (`True.csv`, `Fake.csv`) used for front-page highlights and model development

The app is designed as a **decision-support tool**. It should not be used as the sole source of truth for fact-checking.

## Features

- Paste any news text and get a prediction:
  - `Not A Fake News`
  - `Fake News`
- Confidence display (when model probabilities are available)
- Newspaper-style Streamlit interface
- Sample headlines loaded from local datasets (`True.csv` and `Fake.csv`)

## Project Structure

- `app.py` — Streamlit application entry point
- `news.ipynb` — model training/evaluation notebook
- `model.joblib` — serialized classifier
- `vectorizer.joblib` — serialized TF-IDF vectorizer
- `True.csv` — examples of true news
- `Fake.csv` — examples of fake news
- `requirements.txt` — Python dependencies
- `.devcontainer/` — Codespaces/devcontainer setup (auto-runs Streamlit)

## Tech Stack

- Python 3
- Streamlit
- pandas
- scikit-learn
- joblib

## Quick Start

### 1) Clone and enter the project

```bash
git clone https://github.com/tjadnan1520/verifyNEWS.git
cd verifyNEWS
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (typically `http://localhost:8501`).

## How Prediction Works

1. Input text is cleaned in `wordopt()`:
   - lowercasing
   - removing URLs, punctuation, HTML fragments, newlines, and alphanumeric tokens
2. Cleaned text is transformed by `vectorizer.joblib` (TF-IDF)
3. `model.joblib` predicts class probabilities or class label
4. App returns:
   - `Not A Fake News` when class-1 probability is at least 0.4
   - otherwise `Fake News`

> Note: The threshold and class mapping are based on the current implementation in `app.py`.

## Data and Sample Highlights

- The sidebar prediction works independently from sample headlines.
- Front-page highlights are built from `True.csv` and `Fake.csv`.
- If either CSV is missing, the app still runs, but sample highlights are not shown.

## Reproducing / Updating the Model

Use `news.ipynb` to retrain and export artifacts.

From the notebook flow:

- Data split with `train_test_split`
- Text features via `TfidfVectorizer`
- Classifier: `LogisticRegression(max_iter=500, C=0.5)`
- Export artifacts:
  - `joblib.dump(LR, "model.joblib")`
  - `joblib.dump(vectorization, "vectorizer.joblib")`

After retraining, keep the exported files in the project root so `app.py` can load them.

## Requirements

Install dependencies listed in `requirements.txt`:

- streamlit
- joblib
- pandas
- scikit-learn

## Troubleshooting

- **Error: missing model files**
  - Ensure `model.joblib` and `vectorizer.joblib` exist in the project root.
- **No sample headlines displayed**
  - Ensure both `True.csv` and `Fake.csv` are present and readable.
- **Import errors**
  - Reinstall dependencies with `pip install -r requirements.txt`.

## Dev Container / Codespaces

This repo includes `.devcontainer/devcontainer.json`.

In Codespaces/devcontainer:

- dependencies are installed automatically
- Streamlit starts automatically via post-attach command
- port `8501` is forwarded and labeled as the application

## Limitations and Responsible Use

- Predictions are probabilistic, not factual verification.
- Model quality depends on training data quality and domain coverage.
- The app may be less reliable on out-of-domain, very short, or highly novel text.

Always combine results with trusted sources and manual verification.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Open a pull request describing:
   - what changed
   - why it changed
   - how to test

## Acknowledgment

Built as an educational/project implementation of machine-learning-based fake news detection with a simple newsroom-style interface.
