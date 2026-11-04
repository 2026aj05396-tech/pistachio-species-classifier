"""Streamlit demo: Pistachio species classification (Kirmizi vs Siirt)."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Pistachio Classifier", page_icon="🌰", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_resource
def load_encoder():
    return joblib.load(MODEL_DIR / "label_encoder.joblib")


def score(model, X, y_true):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, proba, {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
    }


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌰 Pistachio Classifier")
st.sidebar.write("Predict whether a pistachio is **Kirmizi** or **Siirt** "
                 "from 28 image-derived shape and colour features.")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
st.sidebar.caption("Models are pre-trained and loaded from disk. "
                   "Nothing is retrained on your upload.")
sample = BASE / "test_data.csv"
if sample.exists():
    st.sidebar.download_button("Download sample test_data.csv",
                               sample.read_bytes(), "test_data.csv", "text/csv")

# ------------------------------------------------------------------- main
st.title("Pistachio Species Classification")
st.write("Upload a CSV containing the 28 feature columns **and** the "
         f"`{TARGET}` column, pick a model, and see how it performs.")

if uploaded is None:
    st.info("👈 Upload a test CSV in the sidebar to begin "
            "(you can download a sample from there).")
    st.stop()

data = pd.read_csv(uploaded)
model = load_model(model_name)
le = load_encoder()
features = list(model.feature_names_in_)

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing required column(s): {', '.join(missing)}")
    st.stop()

unknown = set(data[TARGET].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unexpected labels in `{TARGET}`: {unknown}. Expected {list(le.classes_)}.")
    st.stop()

X = data[features]
y = le.transform(data[TARGET])

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"{data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(20))

pred, proba, metrics = score(model, X, y)

st.subheader(f"Results: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=le.classes_, cmap="YlGnBu", ax=ax, colorbar=False)
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y, proba, name=model_name, ax=ax,
                                     pos_label=1)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    st.pyplot(fig)

st.markdown("**Classification report**")
report = classification_report(y, pred, target_names=le.classes_, output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("All models on this data")
rows = []
for name in MODEL_FILES:
    _, _, m = score(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(comparison.style.format("{:.4f}")
             .highlight_max(axis=0, color="#c6efce"))
st.caption(f"Best model by MCC: **{comparison['MCC'].idxmax()}**")
