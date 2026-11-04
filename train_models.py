"""Train 5 classifiers on the Pistachio (28 features) dataset and save them."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 55
TARGET = "Class"

df = pd.read_csv("data/pistachio_full.csv")
le = LabelEncoder()
y = le.fit_transform(df[TARGET])          # Kirmizi -> 0, Siirt -> 1
X = df.drop(columns=[TARGET])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# test set is saved with original string labels, for upload in the app
test_df = X_te.copy()
test_df[TARGET] = le.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=2.0, probability=True, random_state=SEED),
}

rows = []
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "MCC": matthews_corrcoef(y_te, pred),
    })
    joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib")

joblib.dump(le, "model/label_encoder.joblib")
res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
print(f"train={len(X_tr)} test={len(X_te)}  positive class = {le.classes_[1]}")
print(res.to_string(index=False))
