# 🛡️ Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project detects fraudulent credit card transactions in a severely imbalanced dataset (0.17% fraud rate). It compares a supervised approach (Logistic Regression, with and without SMOTE) against an unsupervised approach (Isolation Forest), tunes the decision threshold against a real operational constraint, and uses SHAP to make individual predictions explainable. A Streamlit app provides single-transaction and batch prediction on top of the trained model.

**Live demo:** https://creditcardfraudddetection.streamlit.app/

---

## 🧠 The Problem

Fraud makes up less than 0.2% of transactions in this dataset. A model that predicts "genuine" for everything would be 99.8% accurate and completely useless — so accuracy is not used anywhere in this project as an evaluation metric. Instead, the core question this project answers is: **how do you catch rare fraud cases without burying a review team in false alarms?**

---

## 📊 Model Comparison

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.85 | 0.56 | 0.67 | — | — |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 0.11 | 0.979 | 0.740 |
| Isolation Forest (unsupervised) | 0.33 | 0.34 | 0.33 | — | — |

**Key finding:** SMOTE raises fraud recall from 0.56 → 0.92 (catches far more real fraud) but collapses precision from 0.85 → 0.06 (1,440 false alarms vs. 10). This is a direct precision-recall tradeoff with a real operational cost — at the default 0.5 threshold, this volume of false positives would overwhelm any realistic fraud review team.

Rather than accept the default threshold, the notebook selects an operating point based on a stated review-capacity constraint (~300 flagged transactions/day), converting the model's raw probability output into an actual deployment decision rather than a modeling artifact.

Isolation Forest, trained with **no fraud labels at all**, underperforms both supervised approaches here — expected, since it can only detect generic outliers rather than learned fraud patterns. Its real value would be in a cold-start scenario with no historical fraud labels (e.g., an entirely new fraud pattern), not as a replacement for the supervised model.

---

## 🔍 Explainability

SHAP (bar, beeswarm, and waterfall plots) is used to explain individual fraud predictions from the SMOTE-trained model, so a flagged transaction can be justified by its specific contributing features rather than treated as a black-box output.

*(Add a saved SHAP plot image here, e.g. `assets/shap_waterfall.png`)*

---

## 🚀 App Features

- 💳 Single transaction prediction
- 📂 Batch prediction via CSV upload
- 📊 Fraud probability score
- 📈 Prediction summary chart
- 📋 Dashboard metrics
- 🔍 Filter to fraud-only transactions
- 📥 Download prediction results
- 🎨 Interactive Streamlit UI

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Isolation Forest, StandardScaler)
- Imbalanced-learn (SMOTE)
- SHAP (model explainability)
- Streamlit (web app)
- Joblib (model persistence)

---

## 📁 Project Structure

```
CreditCardFraudDetection/
│── app.py                      # Streamlit web app
│── fraud_detection_model.pkl   # Trained Logistic Regression + SMOTE model
│── scaler.pkl                  # StandardScaler fit on training data
│── requirements.txt
│── notebooks/
│   └── 02_modeling.ipynb       # Full EDA → modeling → threshold tuning → SHAP notebook
│── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the analysis and retrain the model:

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

---

## 📷 Screenshots

*(Add screenshots of the deployed app here — single prediction view, batch upload view, and dashboard metrics.)*

---

## 📌 Notes on Methodology

- The train/test split happens **before** SMOTE is applied, and SMOTE is fit only on the training fold — the test set is never resampled, avoiding data leakage in evaluation.
- PR-AUC (Average Precision) is reported alongside ROC-AUC, since ROC-AUC is known to look overly optimistic under severe class imbalance.
- The classification threshold is treated as a business decision, not a modeling default — see Section 8 of the notebook for the threshold-selection method.

---

## 👨‍💻 Author

**Madhur**