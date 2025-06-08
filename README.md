# 🧠 Machine Learning Challenge for an Internship

This repository contains the solution developed during a machine learning internship competition. The challenge was hosted on [Kaggle](https://www.kaggle.com/competitions/ai-for-alpha-test-challenge/overview) and involved building a binary classifier from a dataset with **no contextual information**. The source of the data was unknown, though potentially financial. The only known requirement was to predict a **highly imbalanced binary target**, with performance evaluated using the **F1 score**.

---

## 📁 Repository Structure

```

.
├── data/                # Directory for raw and processed data (not included)
├── notebooks/           # Jupyter notebooks for exploratory analysis and development
├── src/                 # Core modules: feature engineering, modeling, utils
├── output/              # Saved models, predictions, plots
├── requirements.txt     # Python dependencies
└── README.md            # Project overview

```

---

## 🧠 Methodology

### 🧹 Data Understanding & Preparation
- Standard preprocessing: normalization, handling missing values
- Identified class imbalance and implemented stratified sampling

### 📊 Feature Engineering

- **Statistical Features**: Mean, median, variance, skewness, kurtosis 
- **Outlier detection**: Detect and search for information on outliers
- **Mutual Information**: Assessed dependency between features and the target
- **XGBoost Feature Importance**: Guided feature selection and pruning
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA)
  - **Autoencoders**: Used to capture non-linear low-dimensional structure

### 🔍 Unsupervised Learning

- **Clustering for Feature Augmentation**:
  - **K-Means**: to identify global patterns
  - **DBSCAN**: to detect dense subgroups and outliers
- Added cluster labels as additional features

---

## 🧪 Modeling & Optimization

- **Primary model**: [XGBoost](https://xgboost.readthedocs.io/) classifier
- **Hyperparameter tuning** using the `scipy.optimize` library
- **Transformers** (based on self-attention) were also tested, but did not yield satisfactory results in this context
- Model evaluation used cross-validation and custom thresholding for F1 score

---

## 🧰 Tools & Libraries

- `xgboost`
- `scikit-learn`
- `scipy.optimize`
- `tensorflow / keras` (autoencoders)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` clustering and metrics


---

## ✍️ Author

**Martin Pasche**
Developed during an internship postulation. Focused on data-driven feature engineering, dimensionality reduction, clustering, and model optimization.




