import streamlit as st
import joblib
import re
import string
import spacy
import pandas as pd
from newspaper import Article

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------- Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ---------- Load models ----------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    log_model = joblib.load("logistic_model.pkl")
    nb_model = joblib.load("naive_bayes_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    lgb_model = joblib.load("lightgbm_model.pkl")
    return vectorizer, log_model, nb_model, xgb_model, lgb_model

vectorizer, log_model, nb_model, xgb_model, lgb_model = load_models()

# ---------- Load model accuracies ----------
accuracies = joblib.load("model_accuracies.pkl")  # Must be precomputed and saved

# ---------- Page config ----------
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.markdown("<h1 style='text-align:center;'>Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Accuracy Comparison Section ----------
st.markdown("### 📊 Model Accuracies (on Test Set)")
for model, acc in accuracies.items():
    st.markdown(f"**{model}:** {acc * 100:.2f}%")

# Highlight best performers
if (accuracies["Logistic Regression"] > accuracies["XGBoost"] and
    accuracies["Logistic Regression"] > accuracies["LightGBM"] and
    accuracies["Naive Bayes"] > accuracies["XGBoost"] and
    accuracies["Naive Bayes"] > accuracies["LightGBM"]):
    st.success("✅ Logistic Regression and Naive Bayes perform better than XGBoost and LightGBM on the test set.")
else:
    st.info("ℹ️ Tree-based models (XGBoost, LightGBM) may perform better on certain datasets.")

# Optional: Accuracy bar chart
st.markdown("### 📉 Accuracy Comparison Chart")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
st.bar_chart(acc_df.set_index("Model"))

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Manual input prediction ----------
st.subheader("📝 Test a News Article Manually")
with st.form("manual_input_form"):
    title = st.text_input("Enter News Title")
    content = st.text_area("Enter News Content", height=200)
    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    if not title or not content:
        st.warning("Please enter both the title and the content.")
    else:
        full_text = title + " " + content
        cleaned = clean_text(full_text)
        vec = vectorizer.transform([cleaned])

        preds = {
            "Logistic Regression": log_model.predict(vec)[0],
            "Naive Bayes": nb_model.predict(vec)[0],
            "XGBoost": xgb_model.predict(vec)[0],
            "LightGBM": lgb_model.predict(vec)[0]
        }

        label_map = {0: "Fake", 1: "Real"}
        st.markdown("### Model Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Logistic Regression:** {label_map[preds['Logistic Regression']]}")
            st.markdown(f"**Naive Bayes:** {label_map[preds['Naive Bayes']]}")
        with col2:
            st.markdown(f"**XGBoost:** {label_map[preds['XGBoost']]}")
            st.markdown(f"**LightGBM:** {label_map[preds['LightGBM']]}")

        all_vals = list(preds.values())
        if all_vals.count(all_vals[0]) == len(all_vals):
            st.success(f"All models agree: {label_map[all_vals[0]]}")
        elif all_vals.count(0) == 2:
            st.info("Two models say Fake, two say Real.")
        else:
            st.warning("Disagreement among models.")

# ---------- URL input prediction ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🌐 Test Live News from a URL")

url = st.text_input("Enter a News Article URL")
if st.button("Fetch and Predict"):
    if not url:
        st.warning("Please provide a valid URL.")
    else:
        try:
            article = Article(url)
            article.download()
            article.parse()
            url_title = article.title
            url_text = article.text

            if not url_text.strip():
                st.warning("Could not extract text from this URL.")
            else:
                full_text = url_title + " " + url_text
                cleaned = clean_text(full_text)
                vec = vectorizer.transform([cleaned])

                preds = {
                    "Logistic Regression": log_model.predict(vec)[0],
                    "Naive Bayes": nb_model.predict(vec)[0],
                    "XGBoost": xgb_model.predict(vec)[0],
                    "LightGBM": lgb_model.predict(vec)[0]
                }

                label_map = {0: "Fake", 1: "Real"}

                st.markdown("**Extracted Title:**")
                st.info(url_title)

                st.markdown("### Model Predictions")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Logistic Regression:** {label_map[preds['Logistic Regression']]}")
                    st.markdown(f"**Naive Bayes:** {label_map[preds['Naive Bayes']]}")
                with col2:
                    st.markdown(f"**XGBoost:** {label_map[preds['XGBoost']]}")
                    st.markdown(f"**LightGBM:** {label_map[preds['LightGBM']]}")

                all_vals = list(preds.values())
                if all_vals.count(all_vals[0]) == len(all_vals):
                    st.success(f"All models agree: {label_map[all_vals[0]]}")
                elif all_vals.count(0) == 2:
                    st.info("Two models say Fake, two say Real.")
                else:
                    st.warning("Disagreement among models.")
        except Exception as e:
            st.error(f"Failed to fetch article: {e}")
