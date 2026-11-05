# Pistachio Species Classification (Kirmizi vs Siirt)

ML Assignment 2 – M.Tech (AIML), BITS Pilani WILP

## a. Problem statement

Kirmizi and Siirt are two pistachio species that sell in different markets, so sorting them correctly after harvest matters commercially. The task is to build a binary classifier that predicts the species of a pistachio from 28 numeric features extracted from its image (shape, size and colour statistics). Five classification algorithms are trained on the same data, compared on six metrics, and exposed through a Streamlit web app.

## b. Dataset description

- **Name:** Pistachio Dataset (28 Features version)
- **Source:** Kaggle / Murat Koklu's dataset page (https://www.muratkoklu.com/datasets/)
- **Instances:** 2,148 (Kirmizi 1,232 · Siirt 916)
- **Features:** 28, all numeric, plus 1 target column `Class`
  - 12 morphological: Area, Perimeter, Major_Axis, Minor_Axis, Eccentricity, Eqdiasq, Solidity, Convex_Area, Extent, Aspect_Ratio, Roundness, Compactness
  - 4 shape factors: Shapefactor_1 … Shapefactor_4
  - 12 colour features: mean, standard deviation, skewness and kurtosis for each of the R-R, R-G, R-B channels
- **Data quality:** no missing values and no duplicate rows.
- **Preprocessing:** stratified 80/20 train–test split (random_state = 55 → 1,718 train / 430 test); `StandardScaler` fitted on the training data inside each model pipeline. Label encoding: Kirmizi = 0, Siirt = 1 (Siirt is the positive class for precision/recall/F1/AUC).
- **Citation:** Ozkan IA., Koklu M., Saracoglu R. (2021). *Classification of Pistachio Species Using Improved K-NN Classifier.* Progress in Nutrition, 23(2). DOI: 10.23751/pn.v23i2.9686

## c. GitHub repository link

https://github.com/2026aj05396-tech/pistachio-species-classifier

Live app: https://pistachio-species-classifier-5xuvmt7mqdjnklegyxnwp4.streamlit.app/

Repository layout:

```
├── app.py               # Streamlit app
├── train_models.py      # trains the 5 models, saves them and test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv        # 430 held-out rows (upload this in the app)
├── data/pistachio_full.csv
└── model/               # saved pipelines (.joblib), label encoder, metrics_summary.csv
```

## d. Models used

All five models were trained on the same split and evaluated on the same 430-row test set. Each is a scikit-learn Pipeline (StandardScaler → classifier).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9116 | 0.9748 | 0.9096 | 0.8798 | 0.8944 | 0.8188 |
| Decision Tree | 0.8558 | 0.8870 | 0.8418 | 0.8142 | 0.8278 | 0.7041 |
| KNN | 0.9000 | 0.9506 | 0.8846 | 0.8798 | 0.8822 | 0.7953 |
| Naive Bayes | 0.8884 | 0.9528 | 0.8426 | 0.9071 | 0.8737 | 0.7756 |
| Support Vector Machine | 0.9372 | 0.9852 | 0.9239 | 0.9290 | 0.9264 | 0.8717 |

Hyperparameters: Logistic Regression (max_iter=1000); Decision Tree (max_depth=6, min_samples_leaf=5); KNN (k=7); Gaussian Naive Bayes (defaults); SVM (RBF kernel, C=2, probability=True).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy of 0.9116 and AUC of 0.9748, making it the second-best model on both metrics. The confusion matrix is 231/16/22/161, meaning it missed 22 Siirt cases (recall of 0.8798). Since a linear model performs this well, the classes appear to be almost linearly separable. |
| Decision Tree | The weakest performer across all metrics, with an accuracy of 0.8558, AUC of 0.8870, and MCC of 0.7041. I limited the tree depth to 6 and set the minimum leaf size to 5. A single tree gives coarser probability estimates, which likely brings down its AUC. |
| KNN | Achieved an accuracy of 0.9000 and AUC of 0.9506. Being distance-based, it is likely sensitive to feature scaling (the features here are standardised). Despite being a non-linear model, it performs slightly worse than Logistic Regression, and the results also depend on the choice of k=7. |
| Naive Bayes | Achieved the highest recall among the non-SVM models (0.9071), but had the lowest precision after the Decision Tree (0.8426), meaning it tends to over-predict Siirt. The independence assumption is not ideal here because some features are strongly correlated, for example Area and Convex_Area (correlation 0.94). |
| Support Vector Machine | The best-performing model across all six metrics, with an accuracy of 0.9372, AUC of 0.9852, F1 of 0.9264, and MCC of 0.8717. The RBF kernel probably helps it capture the small non-linear patterns in the data, while keeping the classification errors balanced (14 and 13 misclassifications per class). |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the best score on every metric, and its MCC of 0.8717 is clearly ahead of the runner-up, Logistic Regression (0.8188). MCC is a useful tiebreaker here because the classes are slightly imbalanced (about 57% Kirmizi, 43% Siirt). |

## How to run

```bash
pip install -r requirements.txt
python train_models.py        # optional – retrains and overwrites model/ and test_data.csv
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar, choose a model and view the metrics, confusion matrix and classification report. The app loads the saved models and never retrains.
