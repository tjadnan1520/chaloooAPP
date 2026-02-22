from pathlib import Path
import re
import string

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
VECTORIZER_PATH = BASE_DIR / "vectorizer.joblib"


def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_sample_news() -> pd.DataFrame:
    true_path = BASE_DIR / "True.csv"
    fake_path = BASE_DIR / "Fake.csv"
    if not true_path.exists() or not fake_path.exists():
        return pd.DataFrame(columns=["headline", "news", "label"])

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    true_title_col = "title" if "title" in df_true.columns else "text"
    fake_title_col = "title" if "title" in df_fake.columns else "text"
    true_news_col = "text" if "text" in df_true.columns else true_title_col
    fake_news_col = "text" if "text" in df_fake.columns else fake_title_col

    true_items = (
        df_true[[true_title_col, true_news_col]]
        .rename(columns={true_title_col: "headline", true_news_col: "news"})
        .astype(str)
    )
    true_items["headline"] = true_items["headline"].str.strip()
    true_items["news"] = true_items["news"].str.strip()
    true_items = true_items[
        (true_items["headline"].str.len() > 10) & (true_items["news"].str.len() > 50)
    ].head(10)

    fake_items = (
        df_fake[[fake_title_col, fake_news_col]]
        .rename(columns={fake_title_col: "headline", fake_news_col: "news"})
        .astype(str)
    )
    fake_items["headline"] = fake_items["headline"].str.strip()
    fake_items["news"] = fake_items["news"].str.strip()
    fake_items = fake_items[
        (fake_items["headline"].str.len() > 10) & (fake_items["news"].str.len() > 50)
    ].head(10)

    interleaved = []
    for index in range(10):
        if index < len(true_items):
            interleaved.append(
                {
                    "headline": true_items.iloc[index]["headline"],
                    "news": true_items.iloc[index]["news"],
                    "label": "Verified",
                }
            )
        if index < len(fake_items):
            interleaved.append(
                {
                    "headline": fake_items.iloc[index]["headline"],
                    "news": fake_items.iloc[index]["news"],
                    "label": "Likely Fake",
                }
            )

    return pd.DataFrame(interleaved)


def predict_news(text: str, model, vectorizer):
    testing_news = {"text": [text]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)

    confidence = None
    threshold = 0.4  # Lower threshold for 'Not A Fake News'
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(new_xv_test)[0]
        confidence = float(max(probs) * 100)
        # Assume class 1 is 'Not A Fake News', class 0 is 'Fake News'
        if probs[1] >= threshold:
            return "Not A Fake News", confidence
        else:
            return "Fake News", confidence
    else:
        pred_value = model.predict(new_xv_test)[0]
        try:
            pred = int(pred_value)
        except (ValueError, TypeError):
            pred = 0 if str(pred_value).strip().lower() in {"0", "fake", "fake news"} else 1
        if pred == 1:
            return "Not A Fake News", confidence
        return "Fake News", confidence


st.set_page_config(
    page_title="The Daily Verifier",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    .paper-title {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 0;
        color: #000000;
    }
    .paper-subtitle {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        color: #000000;
    }
    .news-card {
        background: #ffffff;
        border: 2px solid #87CEEB;
        border-radius: 0;
        padding: 16px;
        margin-bottom: 14px;
        font-family: 'Georgia', serif;
        color: #000000;
    }
    .section-head {
        font-family: 'Georgia', serif;
        font-size: 1.3rem;
        font-weight: 700;
        border-bottom: 2px solid #87CEEB;
        padding-bottom: 4px;
        margin-top: 8px;
        margin-bottom: 10px;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='paper-title'>THE DAILY VERIFIER</div>", unsafe_allow_html=True)
st.markdown("<div class='paper-subtitle'>Independent Analysis Desk · AI News Screening Edition</div>", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error("Missing model files. Make sure model.joblib and vectorizer.joblib exist in this folder.")
    st.stop()

model, vectorizer = load_artifacts()

with st.sidebar:
    st.header("Fact Check Desk")
    st.write("Paste a news article or paragraph to verify whether it is likely fake or not.")
    user_input = st.text_area("News text", height=250, placeholder="Paste news content here...")
    check_btn = st.button("Check News", use_container_width=True)

    if check_btn:
        if not user_input.strip():
            st.warning("Please enter some news text first.")
        else:
            label, confidence = predict_news(user_input, model, vectorizer)
            if label == "Not A Fake News":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            if confidence is not None:
                st.progress(min(max(confidence / 100, 0.0), 1.0))
                st.caption(f"Model confidence: {confidence:.2f}%")

st.markdown("<div class='section-head'>Front Page Highlights</div>", unsafe_allow_html=True)

samples = load_sample_news()
if samples.empty:
    st.info("Sample headlines are unavailable. Add True.csv/Fake.csv to show front-page highlights.")
else:
    st.caption("Click any headline to view the full news content.")
    for _, row in samples.iterrows():
        with st.expander(f"{row['headline']}"):
            st.markdown(
                f"<div class='news-card'><b>{row['headline']}</b><br><br>{row['news']}</div>",
                unsafe_allow_html=True,
            )

st.caption("Note: This tool provides a model-based prediction and should be used as a supporting signal, not absolute truth.")
