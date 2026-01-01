"""
Machine Learning Models for Stroke Prediction
Comparison of Random Forest, Logistic Regression, XGBoost, and SVM
"""

import pandas as pd
import numpy as np
from config import DATASET_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class StrokePredictor:
    """
    Complete stroke prediction pipeline with multiple ML algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state or DATASET_CONFIG['random_state']
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        df = pd.read_csv(filepath)
        
        # Handle BMI missing values
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        
        # Fill missing BMI with median
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # Drop ID column
        df = df.drop('id', axis=1)
        
        print(f"Dataset loaded: {df.shape}")
        print(f"Missing values after preprocessing:\n{df.isnull().sum()}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def prepare_data(self, filepath: str, test_size: float = 0.2, 
                     balance_method: str = 'smote') -> None:
        """
        Prepare data for training.
        
        Args:
            filepath: Path to CSV file
            test_size: Test set proportion
            balance_method: 'smote', 'undersample', or None
        """
        # Load data
        df = self.load_and_preprocess_data(filepath)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Separate features and target
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        print(f"Train class distribution:\n{self.y_train.value_counts()}")
        
        # Handle class imbalance
        if balance_method == 'smote':
            print("\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"After SMOTE: {self.X_train.shape}")
            print(f"Class distribution after SMOTE:\n{pd.Series(self.y_train).value_counts()}")
        
        elif balance_method == 'undersample':
            print("\nApplying Random Undersampling...")
            rus = RandomUnderSampler(random_state=self.random_state)
            self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            print(f"After undersampling: {self.X_train.shape}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("\nData preparation complete!")
    
    def train_all_models(self) -> None:
        """Train all ML models."""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                **MODEL_CONFIG['random_forest'],
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            print(f"{name} trained successfully!")
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED!")
        print("="*60)
    
    def evaluate_model(self, model_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        return metrics
    
    

    def cross_validate_model(self, model_name: str, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        model = self.models[model_name]
    
        scores = {
            'cv_accuracy': cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv, scoring='accuracy').mean(),
            'cv_f1': cross_val_score(model, self.X_train, self.y_train, 
                                cv=cv, scoring='f1').mean(),
            'cv_recall': cross_val_score(model, self.X_train, self.y_train, 
                                    cv=cv, scoring='recall').mean(),
        }
        return scores

    def compare_models(self) -> pd.DataFrame:
        """Compare all models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = []
        
        for name in self.models.keys():
            metrics = self.evaluate_model(name)
            metrics['model'] = name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        
        print("\n", results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self, save_path: str = None):
        """Plot confusion matrices for all models."""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Stroke', 'Stroke'],
                       yticklabels=['No Stroke', 'Stroke'])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, save_path: str = None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print(f"{model_name} does not support feature importance.")
            return None
    
    def plot_feature_importance(self, model_name: str = 'Random Forest', top_n: int = 10,
                               save_path: str = None):
        """Plot feature importance."""
        feature_importance = self.get_feature_importance(model_name)
        
        if feature_importance is not None:
            plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(top_n)
            
            plt.barh(range(len(top_features)), top_features['importance'], color='#3498db')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTop {top_n} Features for {model_name}:")
            print(top_features.to_string(index=False))
    
    def save_best_model(self, model_name: str, filepath: str = 'best_model.pkl'):
        """Save the best model and preprocessing objects."""
        save_dict = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(save_dict, filepath)
        print(f"\n{model_name} saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str = 'best_model.pkl'):
        """Load a saved model."""
        loaded = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return loaded


def run_complete_pipeline(filepath: str, save_dir: str = 'models'):
    """
    Run the complete ML pipeline.
    
    Args:
        filepath: Path to the CSV file
        save_dir: Directory to save models and plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = StrokePredictor(random_state=42)
    
    # Prepare data
    predictor.prepare_data(filepath, test_size=0.2, balance_method='smote')
    
    # Train all models
    predictor.train_all_models()
    
    # Compare models
    results_df = predictor.compare_models()
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    predictor.plot_confusion_matrices(save_path=f'{save_dir}/confusion_matrices.png')
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    predictor.plot_roc_curves(save_path=f'{save_dir}/roc_curves.png')
    
    # Feature importance for tree-based models
    for model_name in ['Random Forest', 'XGBoost']:
        print(f"\nGenerating feature importance for {model_name}...")
        predictor.plot_feature_importance(
            model_name=model_name,
            top_n=10,
            save_path=f'{save_dir}/feature_importance_{model_name.replace(" ", "_").lower()}.png'
        )
    
    # Save best model (based on F1 score best for imbalanced data)
    best_model = results_df.loc[results_df['f1_score'].idxmax(), 'model']
    print(f"\n{('='*60)}")
    print(f"BEST MODEL: {best_model}")
    print(f"{'='*60}")
    
    predictor.save_best_model(best_model, filepath=f'{save_dir}/best_model.pkl')
    
    # Save results to CSV
    results_df.to_csv(f'{save_dir}/model_comparison.csv', index=False)
    print(f"\nResults saved to {save_dir}/model_comparison.csv")
    
    return predictor, results_df


if __name__ == "__main__":
    # Example usage
    predictor, results = run_complete_pipeline('healthcaredatasetstrokedata.csv', save_dir='models')