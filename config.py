"""
Configuration file for Stroke Prediction Project
Centralized settings for data processing, model training, and evaluation
"""

# Dataset Configuration
DATASET_CONFIG = {
    'filepath': 'healthcaredatasetstrokedata.csv',
    'target_column': 'stroke',
    'id_column': 'id',
    'test_size': 0.2,
    'random_state': 42
}

# Feature Configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'age',
        'avg_glucose_level',
        'bmi'
    ],
    
    'categorical_features': [
        'gender',
        'hypertension',
        'heart_disease',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status'
    ],
    
    'binary_features': [
        'hypertension',
        'heart_disease'
    ]
}

# Data Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'handle_missing_bmi': 'median',  # Options: 'median', 'mean', 'drop'
    'balance_method': 'smote',  # Options: 'smote', 'undersample', None
    'scaling_method': 'standard',  # Options: 'standard', 'minmax', 'robust'
}

# Model Hyperparameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42
    },
    
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
        # scale_pos_weight will be calculated dynamically
    },
    
    'logistic_regression': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'random_state': 42
    },
    
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    }
}

# Cross-Validation Configuration
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc'
]

# Visualization Configuration
PLOT_CONFIG = {
    'figure_size': (12, 6),
    'style': 'whitegrid',
    'color_palette': {
        'no_stroke': '#2ecc71',
        'stroke': '#e74c3c',
        'primary': '#3498db'
    },
    'dpi': 300,
    'save_format': 'png'
}

# Risk Thresholds for UI
RISK_THRESHOLDS = {
    'low': 0.05,       # < 5%
    'moderate': 0.15,  # 5-15%
    'high': 0.30,      # 15-30%
    'very_high': 1.0   # > 30%
}

# Health Recommendations Thresholds
HEALTH_THRESHOLDS = {
    'bmi': {
        'underweight': 18.5,
        'normal': 25.0,
        'overweight': 30.0,
        'obese': 40.0
    },
    'glucose': {
        'normal': 100,
        'prediabetes': 140,
        'diabetes': 200
    },
    'age_risk': {
        'low': 40,
        'moderate': 50,
        'high': 65
    }
}

# File Paths
PATHS = {
    'plots_dir': 'plots',
    'models_dir': 'models',
    'best_model': 'models/best_model.pkl',
    'model_comparison': 'models/model_comparison.csv'
}

# Streamlit UI Configuration
UI_CONFIG = {
    'page_title': 'Stroke Risk Prediction',
    'page_icon': '🏥',
    'layout': 'wide',
    'age_range': (0, 100),
    'glucose_range': (50.0, 300.0),
    'bmi_range': (10.0, 60.0)
}

# Feature Descriptions (for UI tooltips)
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'gender': 'Biological sex assigned at birth',
    'hypertension': 'High blood pressure (≥140/90 mmHg)',
    'heart_disease': 'History of heart disease',
    'ever_married': 'Marital status',
    'work_type': 'Type of employment',
    'Residence_type': 'Urban or rural residence',
    'avg_glucose_level': 'Average blood glucose level in mg/dL',
    'bmi': 'Body Mass Index (weight in kg / height in m²)',
    'smoking_status': 'Current or past smoking habits'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Feature Engineering Options (for future enhancements)
FEATURE_ENGINEERING = {
    'create_age_groups': True,
    'create_bmi_categories': True,
    'create_glucose_categories': True,
    'interaction_features': False  # Can be enabled for advanced modeling
}

# Model Selection Criteria
MODEL_SELECTION = {
    'primary_metric': 'f1_score',  # Best for imbalanced data
    'minimum_recall': 0.70,  # Prefer high recall for stroke detection
    'use_cross_validation': True
}

# Export/Import Configuration
EXPORT_CONFIG = {
    'save_preprocessing_pipeline': True,
    'save_feature_importance': True,
    'save_model_metrics': True,
    'model_format': 'pkl'  # Options: 'pkl', 'joblib'
}