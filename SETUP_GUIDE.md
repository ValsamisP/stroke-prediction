# 🚀 Stroke Prediction System - Complete Setup Guide

This guide will walk you through setting up and running the entire Stroke Prediction & Prevention System from scratch.

---

##  Prerequisites

### Required Software
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** (Python package installer)
- **Git** (optional, for version control)

### Check Your Python Installation
```bash
python --version
# or
python3 --version
```

---

##  Step 1: Project Setup

### Option A: Manual Setup

1. **Create project directory:**
```bash
mkdir stroke-prediction
cd stroke-prediction
```

2. **Download/create all project files:**
   - `healthcaredatasetstrokedata.csv` (your dataset)
   - `visualization.py`
   - `ml_models.py`
   - `app.py`
   - `main.py`
   - `config.py`
   - `test_predictions.py`
   - `requirements.txt`
   - `README.md`

3. **Verify file structure:**
```bash
stroke-prediction/
├── healthcaredatasetstrokedata.csv
├── visualization.py
├── ml_models.py
├── app.py
├── main.py
├── config.py
├── test_predictions.py
├── requirements.txt
└── README.md
```

---

##  Step 2: Environment Setup

### Option 1: Using Virtual Environment (Recommended)

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n stroke-pred python=3.9

# Activate environment
conda activate stroke-pred
```

---

##  Step 3: Install Dependencies

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib & seaborn (visualization)
- scikit-learn (ML algorithms)
- xgboost (gradient boosting)
- imbalanced-learn (handling imbalanced data)
- streamlit (web interface)
- plotly (interactive visualizations)
- joblib (model serialization)

**Verify installation:**
```bash
pip list
```

---

##  Step 4: Run the Complete Pipeline

### Method 1: Using the Main Script (Easiest)

```bash
python main.py
```

Then select option **1** to run the complete pipeline.

### Method 2: Manual Step-by-Step Execution

#### A. Run Exploratory Data Analysis

```python
python -c "from visualization import run_complete_eda; run_complete_eda('healthcaredatasetstrokedata.csv', 'plots')"
```

**Expected outputs in `plots/` directory:**
- `class_distribution.png`
- `numerical_distributions.png`
- `categorical_distributions.png`
- `correlation_matrix.png`
- `age_analysis.png`

#### B. Train Machine Learning Models

```python
python -c "from ml_models import run_complete_pipeline; run_complete_pipeline('healthcaredatasetstrokedata.csv', 'models')"
```

**Expected outputs in `models/` directory:**
- `best_model.pkl` (trained model + preprocessing objects)
- `model_comparison.csv` (performance metrics)
- `confusion_matrices.png`
- `roc_curves.png`
- `feature_importance_random_forest.png`
- `feature_importance_xgboost.png`

**Expected console output:**
```
Training Random Forest...
Training XGBoost...
Training Logistic Regression...
Training SVM...
```

#### C. Launch Web Application

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 🧪 Step 5: Test the System

### Run Predefined Test Cases

```bash
python test_predictions.py
```

Select option **1** to run batch predictions on 8 diverse test cases.

**Expected output:**
```
CASE: Healthy Young Adult
Patient Profile:
  • Age: 25 years
  • Gender: Female
  • BMI: 22.5
  • Glucose Level: 85.0 mg/dL
  • Hypertension: No
  • Heart Disease: No
  • Smoking Status: never smoked

Prediction Results:
  • Stroke Probability: 2.34%
  • Risk Level: Low Risk
  • Binary Prediction: No Immediate Risk
```

### Interactive Testing

```bash
python test_predictions.py
```

Select option **2** for interactive mode where you can enter custom patient data.

---

##  Step 6: Using the Web Application

### Navigating the Interface

1. **Open the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter patient information in the sidebar:**
   - Demographics (gender, age, marital status, etc.)
   - Health metrics (glucose, BMI, blood pressure, etc.)

3. **Click "Assess Risk"** button

4. **View results:**
   - **Risk gauge:** Visual representation of stroke probability
   - **Risk level:** Color-coded risk classification
   - **Feature importance:** Which factors contribute most to the risk
   - **Personalized recommendations:** Actionable steps to reduce risk

### Example Input

Try this high-risk patient:
- Gender: Male
- Age: 67
- Marital Status: Yes
- Work Type: Private
- Residence: Urban
- Hypertension: Yes (1)
- Heart Disease: Yes (1)
- Glucose: 228.7 mg/dL
- BMI: 36.6
- Smoking: smokes

**Expected Output:**
- Stroke Probability: ~35-45%
- Risk Level: Very High Risk
- Multiple critical recommendations

---

##  Step 7: Understanding the Results

### Model Performance Metrics

**Accuracy:** Overall prediction correctness
- Not the best metric for imbalanced data

**Precision:** Of predicted strokes, how many are actually strokes?
- Lower due to class imbalance

**Recall (Sensitivity):** Of actual strokes, how many did we catch?
- **Most important metric** for healthcare
- Higher is better - we don't want to miss stroke cases

**F1-Score:** Balance between precision and recall
- Best metric for model selection with imbalanced data

**ROC-AUC:** Overall discriminative ability
- Higher is better

### Feature Importance Interpretation

Top features typically include:
1. **Age** (highest importance)
2. **Average glucose level**
3. **BMI**
4. **Hypertension**
5. **Heart disease**

---

##  Troubleshooting

### Common Issues and Solutions

#### Issue 1: Module not found error
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution:**
```bash
pip install -r requirements.txt
```

#### Issue 2: Dataset not found
```
FileNotFoundError: healthcaredatasetstrokedata.csv
```
**Solution:**
Ensure the CSV file is in the same directory as your Python scripts.

#### Issue 3: Model file not found
```
FileNotFoundError: models/best_model.pkl
```
**Solution:**
Run the training pipeline first:
```bash
python main.py
# Select option 1 or 3
```

#### Issue 4: Streamlit command not found
```
streamlit: command not found
```
**Solution:**
```bash
pip install streamlit
# or
python -m streamlit run app.py
```

#### Issue 5: Port already in use
```
Address already in use
```
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

#### Issue 6: Low model performance
**Possible causes:**
- Random seed variation
- Data preprocessing issues
- Class imbalance not properly handled

**Solution:**
- Re-run training with different random seeds
- Check SMOTE is applied correctly
- Review data preprocessing steps

---

##  Advanced Usage

### Customizing Model Parameters

Edit `config.py` to adjust:
- Model hyperparameters
- Data preprocessing settings
- Risk thresholds
- Feature engineering options

Example:
```python
# In config.py
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,  # Increase trees
        'max_depth': 15,      # Deeper trees
        ...
    }
}
```

Then retrain:
```bash
python main.py
# Select option 3 (Model Training only)
```

### Hyperparameter Tuning

For optimal results, implement grid search:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)
```

### Adding New Features

1. Update the dataset with new columns
2. Modify `FEATURE_CONFIG` in `config.py`
3. Update preprocessing in `ml_models.py`
4. Retrain models

---

##  Project Workflow Summary

```
1. Data Exploration (visualization.py)
   ↓
2. Model Training (ml_models.py)
   ↓
3. Model Evaluation & Selection
   ↓
4. Save Best Model
   ↓
5. Deploy via Web App (app.py)
   ↓
6. Make Predictions & Get Recommendations
```

---

##  Daily Usage Workflow

### For Research/Development:
```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run EDA if exploring new data
python -c "from visualization import run_complete_eda; run_complete_eda('data.csv', 'plots')"

# Train models
python main.py  # Option 3

# Test predictions
python test_predictions.py
```

### For End Users:
```bash
# Activate environment
source venv/bin/activate

# Launch web app
streamlit run app.py

# Use the browser interface to assess patients
```

---

## 🎓 Learning Resources

### Understanding the Code

**Start with:**
1. `config.py` - Understand all settings
2. `visualization.py` - Learn EDA techniques
3. `ml_models.py` - Study ML pipeline
4. `app.py` - Explore UI development

**Key Concepts:**
- Class imbalance handling (SMOTE)
- Model evaluation metrics
- Feature importance
- Streamlit app development

### Next Steps

- Implement cross-validation
- Add SHAP values for explainability
- Create ensemble models
- Deploy to cloud (Heroku, AWS, GCP)
- Build REST API
- Add more features to improve predictions

---

##  Data Privacy & Ethics

**Important Reminders:**
- This is an educational tool, not a diagnostic device
- Never store real patient data without proper security
- Always get informed consent
- Follow HIPAA/GDPR guidelines if using real data
- Explain model limitations to users
- Encourage professional medical consultation

---


For questions or issues, review the troubleshooting section or check the main README.md file.