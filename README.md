# Machine Learning Classification Projects — Final Academic Submission Repository

This repository contains multiple machine learning classification projects completed as part of university coursework, including:

- **APS System Failure Prediction (Data Analytics & Big Data)**
- **Loan Approval Prediction (Data Mining)**
- **Naive Bayes Manual Classification Demonstration**

Each project includes complete end-to-end machine learning workflows: data preprocessing, exploratory analysis, feature engineering, model selection, training, evaluation, and performance comparison using real datasets.

---

## 📁 Projects Included

### **1. APS System Failure Prediction using Supervised Learning**
Course: **SEP 6DA3 — Data Analytics and Big Data (Fall 2024)**  
Instructor: **Siqi Zhao**  
Group: **Group 19**

This project predicts air pressure system failures in Scania heavy trucks using supervised ML models.

#### **Objective**
- Preprocess high-dimensional sensor data
- Compare different classifiers
- Handle extreme class imbalance
- Evaluate using ROC/AUC, confusion matrices & metrics

#### **Dataset Overview**
- Training: 60,000 samples (59,000 negative, 1,000 positive)
- Testing: 16,000 samples
- Features: 171 anonymized sensor attributes (including histogram bins)
- Class labels:
  - `pos` → Failure
  - `neg` → Non-failure

#### **Data Preparation & Cleaning**
Steps performed:
1. Replaced `"na"` with `NaN`
2. Dropped columns with >30% missing values
3. Filled remaining NaN with column means
4. Converted label to binary (`1/0`)
5. Feature type conversion → numeric
6. Removed duplicates
7. Low-variance features removed

#### **Feature Engineering & Analysis**
- Class imbalance visualization
- Feature histograms (first 10 features)
- Correlation matrix heatmap
- StandardScaler applied
- PCA reduction to 52 components

#### **Models Implemented**
- Decision Tree
- Logistic Regression (L2 Regularization)
- Random Forest

---

### 🔍 **Decision Tree Results**
**Confusion Matrix:**
- TN: 15255  
- FP: 370  
- FN: 95  
- TP: 280  

**Metrics**
- ROC-AUC: 0.861
- CV ROC-AUC (5 folds): Mean = 0.9826

**Observations**
- Good performance on majority class
- Minority class precision lower due to imbalance

---

### 🔍 **Logistic Regression Results**
Settings:
- C = 0.001
- PCA input

**Confusion Matrix:**
- Strong performance with high recall for positive class

**Metrics**
- Accuracy: 97%
- ROC-AUC: 0.9879
- CV ROC-AUC Mean: 0.9808

**Observations**
- Excellent class separation
- Low precision for minority class due to FP tradeoff

---

### 🔍 **Random Forest Results**
Settings:
- n_estimators = 25
- max_features = 'log2'
- oob_score = True

**Confusion Matrix:**
- TN: 15388
- FP: 237
- FN: 47
- TP: 328

**Metrics**
- Accuracy: 98%
- ROC-AUC: 0.9902
- CV ROC-AUC Mean: 0.99953

**Observations**
- Best performance across all metrics
- Handles imbalanced data more effectively

---

### 📊 **APS Comparative Evaluation**

| Model | ROC-AUC | CV AUC | Accuracy | Precision | Recall | F1 |
|------|---------|--------|----------|-----------|--------|----|
| Decision Tree | 0.86 | 0.98 | 98% | 98% | 97% | 97% |
| Logistic Regression | 0.98 | 0.98 | 98% | 97% | 97% | 97% |
| **Random Forest** | **0.99** | **0.99** | **99%** | **98%** | **98%** | **98%** |

**Best Model:** 🏆 **Random Forest**

#### **Limitations**
- Class imbalance
- PCA interpretation challenges
- Linear model assumptions (LogReg)
- Cost-sensitive evaluation missing

#### **Improvements Suggested**
- SMOTE oversampling
- Hyperparameter optimization
- Threshold adjustments
- Class-weight balancing
- Feature importance analysis

---

## 📁 **2. Loan Approval Prediction (Data Mining Project)**
Course: **SFWRTECH 4DM3 — Data Mining (Winter 2025)**
Instructor: **Jeff Fortuna**

This project trains ML models to predict whether a loan application will be approved.

### Dataset
Includes:
- income, loan amount, term
- cibil_score
- asset values
- loan_status

### Data Preprocessing
Performed:
- Handling missing values
- Feature engineering:
  - debt_to_income
  - total_assets
- Scaling & Encoding
- Train/Test Split

### Models Compared
- SVM (RBF)
- Logistic Regression
- AdaBoost

### Hyperparameter Tuning (CV)
Best Params:
- **SVM:** `C=10`, `kernel=rbf`, `gamma=auto`
- **LogReg:** `C=0.1`, `penalty=l1`, `solver=liblinear`
- **AdaBoost:** `n_estimators=50`, `learning_rate=0.01`

### Results

| Model | Accuracy | AUC | Train(s) | Test(s) | F1 |
|-------|---------|-----|----------|---------|-----|
| SVM | 90.12% | 0.93 | 12.35 | 0.012 | 0.94 |
| LogReg | 88.76% | 0.91 | 1.23 | 0.001 | 0.94 |
| **AdaBoost** | **89.65%** | **0.92** | **3.46** | **0.003** | **0.96** |

**Winner:** 🏆 **AdaBoost**

---

## 📁 **3. Manual Naive Bayes Classifier Demo**

A Python implementation demonstrating Bayesian classification without sklearn.

### Features Used
- Purchase price
- Maintenance cost
- Doors
- Trunk size
- Safety

### Pipeline
- Manual prior computation
- Manual likelihood computation
- Posterior classification
- Error rate calculation

### Purpose
✔ Educational  
✔ Probabilistic intuition  
✔ Zero-library classifier  

---

## 📦 Tools & Technologies
- Python
- pandas
- numpy
- sklearn
- matplotlib
- SMOTE
- PCA

---

## 🎓 Academic Value & Learning Outcomes

Across all projects, the work demonstrates:

✔ Feature engineering & EDA  
✔ Handling imbalanced datasets  
✔ Multi-model benchmarking  
✔ ROC/AUC interpretation  
✔ Confusion matrix reasoning  
✔ Cross-validation tuning  
✔ PCA dimensionality reduction  
✔ Manual ML math understanding  

---

## 🏁 Final Takeaways

- **Random Forest** best for APS failure prediction
- **AdaBoost** best for loan approval ML
- **Naive Bayes** best for interpretability teaching
- **Cross-validation** crucial for parameter tuning
- **ROC/AUC** valuable for imbalanced classes
- **Imbalance handling** is a real-world necessity

---

## 📚 Course Credits

Projects completed under:

- SEP 6DA3 — Data Analytics and Big Data
- SFWRTECH 4DM3 — Data Mining

Institution: **McMaster University, Canada**

---

## 👥 Team Members

- Henis Nakrani
- Ishika Vaghasiya 

---


