# 🧠 Stroke Prediction & Prevention System

A comprehensive machine learning project for predicting stroke risk and providing personalized health recommendations.

---

## Project Overview

This project implements a complete end-to-end machine learning pipeline for stroke prediction, including:

- **Exploratory Data Analysis (EDA)** with comprehensive visualizations  
- **Multiple ML Models** comparison (Random Forest, Logistic Regression, XGBoost, SVM)  
- **Interactive Web Application** for risk assessment  
- **Personalized Recommendations** based on feature importance and user data  

---

## 📁 Project Structure

```
stroke-prediction/
│
├── healthcaredatasetstrokedata.csv # Kaggle stroke dataset
├── visualization.py # EDA plotting functions
├── ml_models.py # Model training and evaluation functions
├── app.py # Streamlit web application
├── config.py # Configuration parameters
├── requirements.txt # Python dependencies
├── Dockerfile # Container configuration
├── README.md 
├── .dockerignore # Docker build exclusions
├── test_predictions.py # Prediction testing utilities
├── 1.ipynb # A few plots for small overview
├── main.py # Main execution script
│
├── plots/
│   ├── class_distribution.png
│   ├── numerical_distributions.png
│   ├── categorical_distributions.png
│   ├── correlation_matrix.png
│   └── age_analysis.png
│
└── models/
    ├── best_model.pkl
    ├── model_comparison.csv
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── feature_importance_random_forest.png
    └── feature_importance_xgboost.png
```

---

## Quick Start (Docker – Recommended)

This is the **recommended and fully reproducible way** to run the project.

### Prerequisites
- Docker Desktop  -- Get Docker (https://docs.docker.com/get-started/get-docker/)
- **Windows users:** WSL 2 must be enabled  

---

### 1️⃣ Build the Docker Image

```bash
docker build -t stroke-ml-app .
```

---

### 2️⃣ Run the Application

```bash
docker run -p 8501:8501 stroke-ml-app
```

---

### 3️⃣ Open in Browser

```
http://localhost:8501
```

✅ No local Python installation required  
✅ Same behavior on Windows, macOS, and Linux  
✅ Identical results across machines  

---

## Reproducibility Guarantees

This project is designed to be **fully reproducible**.

### Environment Reproducibility
- Dockerized Linux environment
- Fixed Python version (`python:3.11-slim`)
- All dependencies **version-pinned** in `requirements.txt`

### Model Reproducibility
- Pre-trained model included: `models/best_model.pkl`
- Model, scaler, encoders, and feature order stored together
- Compatible with `scikit-learn==1.6.1`

### Training Determinism
- Fixed random seeds (`random_state = 42`)
- Deterministic preprocessing pipeline
- Consistent train/test split and SMOTE behavior

### Cross-Platform Consistency
- Same Docker image runs on:
  - Windows
  - macOS
  - Linux
- No OS-specific instructions required

> **Note:** Training the model again may produce *slightly* different metrics due to underlying numerical libraries, but inference results from the provided model are deterministic.

---

## Running Without Docker (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

 This method depends on your local Python environment and is **not guaranteed to be reproducible** across machines.

---

## 📊 Dataset Information

**Source:** Kaggle  
**Samples:** 5,110  
**Features:** 11 (excluding ID)

### Features
- **Demographics:** gender, age, ever_married, work_type, Residence_type  
- **Health Metrics:** hypertension, heart_disease, avg_glucose_level, bmi, smoking_status  
- **Target:** stroke (0 = No, 1 = Yes)

---


## Data Flow

### 1. Data Loading and Preprocessing
```
Raw CSV File
    ↓
Load into pandas DataFrame
    ↓
Remove ID column
    ↓
Handle missing BMI values
    ↓
Encode categorical variables (One-Hot Encoding)
    ↓
Scale numerical features (StandardScaler)
    ↓
Split into train/test (80/20) 
    ↓
Apply SMOTE to training data
    ↓
Ready for model training

```
### 2. Model Training Pipeline
```
Preprocessed Data
    ↓
Train 4 models in parallel:
    ├── Random Forest
    ├── XGBoost
    ├── Logistic Regression
    └── SVM

Evaluate each model:
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1-Score
    └── ROC-AUC
    ↓
Compare models
    ↓
Select best model (based on Recall)
    ↓
Save best model + preprocessing objects

```

### 3. Prediction Pipeline

```

User Input (Web App)
    ↓
Validate input
    ↓
Apply same preprocessing:
    ├── One-Hot Encoding
    ├── Feature Scaling
    └── Feature Ordering
    ↓
Load saved model
    ↓
Make prediction
    ↓
Calculate probability
    ↓
Generate recommendations
    ↓
Display to user
```

---


## Educational Purpose

This project is intended for:
- Learning end-to-end ML workflows  
- Handling imbalanced healthcare datasets  
- Building interpretable ML systems  
- Deploying ML applications with Docker  

---

## Important Disclaimers

- This tool is for **educational purposes only**
- It is **not medical advice**
- Always consult qualified healthcare professionals

---

## Future Enhancements

- [ ] SHAP explainability  
- [ ] Hyperparameter tuning  
- [ ] Cloud deployment   
- [ ] REST API version  

---

## Author

Created by **Panagiotis Valsamis**  
*M.Sc. in Data Science candidate & aspiring Data Scientist*

---

## Support

If something doesn’t work:
1. Ensure Docker is running  
2. Rebuild the image  
3. Verify port 8501 is free  
4. Confirm `models/best_model.pkl` exists  

---

> **This project demonstrates reproducible, production-style machine learning from data to deployment.**
