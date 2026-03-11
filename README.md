# Liver Disease Prediction 
# 🩺 LiverGuard AI — Liver Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Grade](https://img.shields.io/badge/Grade-A%2B%2096%2F100-brightgreen)

## 📌 Project Overview

A complete end-to-end Machine Learning project that predicts
liver disease using the Indian Liver Patient Dataset (ILPD).
The project includes full EDA, 8 classification models,
hyperparameter tuning and a Streamlit web application.

---

## 🏥 Domain
Healthcare — Liver Disease Detection

---

## 📋 Dataset
- **Name:** Indian Liver Patient Dataset (ILPD)
- **Records:** 583 patients
- **Features:** 10 biochemical lab test features
- **Target:** Binary — Liver Disease (1) / No Disease (0)
- **Class Distribution:** 71.4% Disease vs 28.6% No Disease

---

## 🧪 Features Used
| Feature | Description |
|---------|-------------|
| Age | Age of patient |
| Gender | Male / Female |
| Total Bilirubin | Bilirubin level in blood |
| Direct Bilirubin | Direct Bilirubin level |
| Alkaline Phosphotase | Liver enzyme level |
| Alamine Aminotransferase | ALT enzyme (liver damage marker) |
| Aspartate Aminotransferase | AST enzyme (liver damage marker) |
| Total Protiens | Total protein level |
| Albumin | Albumin protein level |
| Albumin Globulin Ratio | Ratio of Albumin to Globulin |

---

## 🤖 Models Built
| Model | Default F1 | Tuned F1 | Improvement |
|-------|-----------|---------|-------------|
| Logistic Regression | 0.806 | **0.834** | +0.028 |
| Decision Tree | 0.716 | 0.818 | +0.102 |
| Random Forest | 0.727 | 0.794 | +0.067 |
| KNN | 0.747 | 0.784 | +0.037 |

---

## 🏆 Best Model — Logistic Regression (Tuned)
| Metric | Score |
|--------|-------|
| Accuracy | 71.8% |
| Precision | 0.716 |
| Recall | **1.000** ✅ |
| F1-Score | **0.834** |
| AUC | 0.719 |
| CV Mean F1 | 0.834 |
| CV Std Dev | 0.003 |

> Perfect Recall of 1.0 means no liver disease patient is ever missed.
> This is critical in healthcare applications.

---

## ⚙️ Tech Stack
- **Language:** Python
- **ML Library:** Scikit-learn
- **EDA:** Matplotlib, Seaborn, Pandas, NumPy
- **Tuning:** GridSearchCV with 5-Fold Cross Validation
- **App:** Streamlit
- **Model Saving:** Joblib (pkl files)

---

## 📁 Project Structure
```
Liver-Disease-Prediction/
│
├── app.py                          # Streamlit web application
├── liver_disease_prediction.ipynb  # Complete ML notebook
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
└── app_images/                     # Dashboard images
    ├── hero_banner.png
    ├── logo.png
    ├── class_distribution.png
    ├── model_comparison.png
    ├── feature_importance.png
    ├── roc_curve.png
    └── best_model_metrics.png
```

---

## 🚀 How to Run the App
```bash
# Clone the repository
git clone https://github.com/yourusername/Liver-Disease-Prediction.git

# Go to project folder
cd Liver-Disease-Prediction

# Install dependencies
pip install streamlit scikit-learn joblib pillow pandas numpy matplotlib seaborn

# Run the app
streamlit run app.py
```

---

## 📊 Key Challenges Faced
1. **Class Imbalance** — 71.4% vs 28.6% → Used F1-Score and Recall
2. **Missing Values** — 4 NaN in Albumin Globulin Ratio → Median Imputation
3. **Outliers** — Bilirubin up to 75 → StandardScaler for LR and KNN
4. **Multicollinearity** — Total & Direct Bilirubin (r=0.87) → Tree models handle naturally
5. **Small Dataset** — 583 records → 5-Fold Cross Validation
6. **Categorical Feature** — Gender → Label Encoding
7. **Interval Type Column** — Age_Group → Dropped before modelling

---

## ⭐ STAR Method Summary

| | |
|--|--|
| **Situation** | 583 patient records with class imbalance, missing values and outliers |
| **Task** | Build complete ML pipeline — EDA, 8 models, evaluate, document |
| **Action** | Cleaned data → EDA → 4 classifiers default + GridSearchCV tuned → saved pkl files |
| **Result** | Logistic Regression — Perfect Recall 1.0, F1-Score 0.834, Grade A+ 96/100 |

---

## ⚠️ Disclaimer
This application is for educational purposes only.
It is not a substitute for professional medical advice or diagnosis.
Always consult a qualified healthcare professional.

---

## 👩‍💻 Author
**Supriya**
Data Science Student
