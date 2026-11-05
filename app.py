"""Pistachio Sorter: compare five classifiers on Kirmizi vs Siirt pistachios.

Models are trained offline by train_models.py and only *loaded* here;
nothing is retrained on the uploaded file.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

HERE = Path(__file__).parent
SAVED = HERE / "model"
LABEL_COL = "Class"
POSITIVE = 1  # Siirt
ZOO = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
SPOTLIGHT = ["Area", "Perimeter", "Major_Axis", "Eccentricity",
             "Roundness", "Aspect_Ratio"]

st.set_page_config(page_title="Pistachio Sorter", page_icon="🥜", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(90deg, #6B8E23 0%, #A3B86C 100%);
           padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem;}
    .hero h1 {color: #fff; margin: 0; font-size: 2rem;}
    .hero p {color: #F3F7E3; margin: .3rem 0 0 0;}
    </style>
    <div class="hero">
      <h1>🥜 Pistachio Sorter</h1>
      <p>Is it a <b>Kirmizi</b> or a <b>Siirt</b>? Five classic ML models, one test set.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def fetch_model(name):
    return joblib.load(SAVED / ZOO[name])


@st.cache_resource
def fetch_labels():
    return joblib.load(SAVED / "label_encoder.joblib")


def evaluate(pipe, feats, truth):
    guess = pipe.predict(feats)
    p_pos = pipe.predict_proba(feats)[:, POSITIVE]
    return guess, p_pos, {
        "Accuracy": accuracy_score(truth, guess),
        "AUC": roc_auc_score(truth, p_pos),
        "Precision": precision_score(truth, guess),
        "Recall": recall_score(truth, guess),
        "F1": f1_score(truth, guess),
        "MCC": matthews_corrcoef(truth, guess),
    }


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Control panel")
    upload = st.file_uploader("1 · Upload test CSV", type="csv")
    chosen = st.selectbox("2 · Pick a model", list(ZOO))
    st.caption("Pre-trained models are loaded from disk. "
               "Your upload is only scored, never used for training.")
    demo_file = HERE / "test_data.csv"
    if demo_file.exists():
        st.download_button("⬇ Get a ready-made test_data.csv",
                           demo_file.read_bytes(), "test_data.csv", "text/csv")
    with st.expander("About the data"):
        st.write("Pistachio Dataset (28 features) by Koklu et al.: shape, "
                 "size and colour statistics from 2,148 pistachio images. "
                 "Positive class for AUC / precision / recall is **Siirt**.")

if upload is None:
    st.info("👈 Start by uploading a test CSV (it must contain the 28 feature "
            f"columns and the `{LABEL_COL}` column). A sample is available in the sidebar.")
    st.stop()

df = pd.read_csv(upload)
pipe = fetch_model(chosen)
labels = fetch_labels()
feature_cols = list(pipe.feature_names_in_)

absent = [c for c in feature_cols + [LABEL_COL] if c not in df.columns]
if absent:
    st.error(f"Your file is missing column(s): {', '.join(absent)}")
    st.stop()

bad_labels = set(df[LABEL_COL].unique()) - set(labels.classes_)
if bad_labels:
    st.error(f"Unexpected values in `{LABEL_COL}`: {bad_labels}. "
             f"Allowed: {list(labels.classes_)}")
    st.stop()

feats = df[feature_cols]
truth = labels.transform(df[LABEL_COL])
guess, p_pos, scores = evaluate(pipe, feats, truth)

tab_perf, tab_all, tab_one = st.tabs(
    ["📊 Performance", "⚖️ Model showdown", "🔍 Inspect one nut"])

# --------------------------------------------------------- tab 1: performance
with tab_perf:
    st.subheader(f"{chosen} on your {len(df)} rows")
    for box, (name, val) in zip(st.columns(6), scores.items()):
        box.metric(name, f"{val:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            truth, guess, display_labels=labels.classes_, cmap="YlGn",
            ax=ax, colorbar=False)
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
    with c2:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(truth, p_pos, name=chosen, ax=ax,
                                         pos_label=POSITIVE,
                                         curve_kwargs={"color": "#6B8E23"})
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        st.pyplot(fig)

    st.markdown("**Classification report**")
    rep = classification_report(truth, guess, target_names=labels.classes_,
                                output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(4))

    out = pd.DataFrame({
        "row": range(len(df)),
        "actual": df[LABEL_COL],
        "predicted": labels.inverse_transform(guess),
        "prob_Siirt": p_pos.round(4),
    })
    out["correct"] = out["actual"] == out["predicted"]
    st.download_button("⬇ Download predictions (CSV)",
                       out.to_csv(index=False), f"predictions_{chosen}.csv",
                       "text/csv")

# ---------------------------------------------------------- tab 2: showdown
with tab_all:
    st.subheader("All five models, same data")
    table = pd.DataFrame(
        [{"ML Model Name": n, **evaluate(fetch_model(n), feats, truth)[2]}
         for n in ZOO]).set_index("ML Model Name")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#DDE8B9"))
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.success(f"Top model by MCC on this file: **{table['MCC'].idxmax()}**")

# ------------------------------------------------------ tab 3: single sample
with tab_one:
    st.subheader("Look at a single pistachio")
    idx = st.slider("Row number in your file", 0, len(df) - 1, 0)
    row = df.iloc[[idx]]
    st.write("Key measurements")
    st.dataframe(row[[c for c in SPOTLIGHT if c in row.columns]],
                 hide_index=True)
    st.write(f"Actual label: **{row[LABEL_COL].iloc[0]}**")

    verdicts = []
    for n in ZOO:
        m = fetch_model(n)
        p = m.predict_proba(row[feature_cols])[0]
        verdicts.append({
            "Model": n,
            "Verdict": labels.classes_[p.argmax()],
            f"P({labels.classes_[0].split('_')[0]})": round(p[0], 3),
            f"P({labels.classes_[1].split('_')[0]})": round(p[1], 3),
        })
    st.dataframe(pd.DataFrame(verdicts).set_index("Model"))
