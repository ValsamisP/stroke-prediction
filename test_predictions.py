"""
Testing and Demo Script for Stroke Prediction System
Run this to test the trained model with sample cases
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List


class StrokePredictionTester:
    """Test the stroke prediction model with various scenarios."""
    
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        """Load the trained model."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.label_encoders = self.model_data['label_encoders']
        self.feature_names = self.model_data['feature_names']
        
    def preprocess_input(self, user_data: Dict) -> np.ndarray:
        """Preprocess user input data."""
        df = pd.DataFrame([user_data])
        
        # Encode categorical features
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        return scaled_data
    
    def predict(self, user_data: Dict) -> tuple:
        """Make prediction for user input."""
        processed_data = self.preprocess_input(user_data)
        probability = self.model.predict_proba(processed_data)[0][1]
        prediction = self.model.predict(processed_data)[0]
        
        return probability, prediction
    
    def get_risk_level(self, probability: float) -> str:
        """Get risk level based on probability."""
        if probability < 0.05:
            return "Low Risk"
        elif probability < 0.15:
            return "Moderate Risk"
        elif probability < 0.30:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def print_prediction(self, name: str, user_data: Dict):
        """Print prediction results nicely formatted."""
        probability, prediction = self.predict(user_data)
        risk_level = self.get_risk_level(probability)
        
        print("\n" + "="*60)
        print(f"CASE: {name}")
        print("="*60)
        
        print("\nPatient Profile:")
        print(f"  • Age: {user_data['age']} years")
        print(f"  • Gender: {user_data['gender']}")
        print(f"  • BMI: {user_data['bmi']:.1f}")
        print(f"  • Glucose Level: {user_data['avg_glucose_level']:.1f} mg/dL")
        print(f"  • Hypertension: {'Yes' if user_data['hypertension'] else 'No'}")
        print(f"  • Heart Disease: {'Yes' if user_data['heart_disease'] else 'No'}")
        print(f"  • Smoking Status: {user_data['smoking_status']}")
        
        print("\nPrediction Results:")
        print(f"  • Stroke Probability: {probability*100:.2f}%")
        print(f"  • Risk Level: {risk_level}")
        print(f"  • Binary Prediction: {'STROKE RISK' if prediction == 1 else 'No Immediate Risk'}")
        
        print("-"*60)


def create_test_cases() -> List[Dict]:
    """Create a variety of test cases."""
    
    test_cases = [
        {
            'name': 'Healthy Young Adult',
            'data': {
                'gender': 'Female',
                'age': 25,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'No',
                'work_type': 'Private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 85.0,
                'bmi': 22.5,
                'smoking_status': 'never smoked'
            }
        },
        {
            'name': 'Middle-Aged with Moderate Risk',
            'data': {
                'gender': 'Male',
                'age': 52,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'Self-employed',
                'Residence_type': 'Urban',
                'avg_glucose_level': 110.0,
                'bmi': 27.8,
                'smoking_status': 'formerly smoked'
            }
        },
        {
            'name': 'High Risk - Multiple Factors',
            'data': {
                'gender': 'Male',
                'age': 67,
                'hypertension': 1,
                'heart_disease': 1,
                'ever_married': 'Yes',
                'work_type': 'Private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 228.7,
                'bmi': 36.6,
                'smoking_status': 'smokes'
            }
        },
        {
            'name': 'Elderly with Good Health',
            'data': {
                'gender': 'Female',
                'age': 72,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'Self-employed',
                'Residence_type': 'Rural',
                'avg_glucose_level': 95.0,
                'bmi': 24.1,
                'smoking_status': 'never smoked'
            }
        },
        {
            'name': 'Young Adult with Diabetes',
            'data': {
                'gender': 'Male',
                'age': 35,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'Govt_job',
                'Residence_type': 'Urban',
                'avg_glucose_level': 210.0,
                'bmi': 32.5,
                'smoking_status': 'never smoked'
            }
        },
        {
            'name': 'Senior with Hypertension Only',
            'data': {
                'gender': 'Female',
                'age': 68,
                'hypertension': 1,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'Private',
                'Residence_type': 'Rural',
                'avg_glucose_level': 105.0,
                'bmi': 28.3,
                'smoking_status': 'never smoked'
            }
        },
        {
            'name': 'Middle-Aged Current Smoker',
            'data': {
                'gender': 'Male',
                'age': 55,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'Private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 98.0,
                'bmi': 26.0,
                'smoking_status': 'smokes'
            }
        },
        {
            'name': 'Very High Risk - All Factors',
            'data': {
                'gender': 'Male',
                'age': 75,
                'hypertension': 1,
                'heart_disease': 1,
                'ever_married': 'Yes',
                'work_type': 'Self-employed',
                'Residence_type': 'Urban',
                'avg_glucose_level': 250.0,
                'bmi': 38.5,
                'smoking_status': 'smokes'
            }
        }
    ]
    
    return test_cases


def run_batch_predictions(tester: StrokePredictionTester, test_cases: List[Dict]):
    """Run predictions on all test cases."""
    print("\n" + "="*60)
    print("STROKE PREDICTION MODEL - BATCH TESTING")
    print("="*60)
    
    results = []
    
    for case in test_cases:
        tester.print_prediction(case['name'], case['data'])
        probability, prediction = tester.predict(case['data'])
        
        results.append({
            'Case': case['name'],
            'Age': case['data']['age'],
            'Probability': f"{probability*100:.2f}%",
            'Risk Level': tester.get_risk_level(probability),
            'Prediction': 'STROKE RISK' if prediction == 1 else 'No Immediate Risk'
        })
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY OF ALL PREDICTIONS")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    probabilities = [float(r['Probability'].strip('%')) for r in results]
    print(f"\nAverage Stroke Probability: {np.mean(probabilities):.2f}%")
    print(f"Minimum Stroke Probability: {np.min(probabilities):.2f}%")
    print(f"Maximum Stroke Probability: {np.max(probabilities):.2f}%")
    print(f"Standard Deviation: {np.std(probabilities):.2f}%")
    
    # Risk level distribution
    print("\nRisk Level Distribution:")
    risk_levels = [r['Risk Level'] for r in results]
    for level in ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']:
        count = risk_levels.count(level)
        print(f"  • {level}: {count} cases ({count/len(results)*100:.1f}%)")


def interactive_mode(tester: StrokePredictionTester):
    """Interactive mode for custom predictions."""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("\nEnter patient information to get a stroke risk assessment.")
    
    try:
        user_data = {
            'gender': input("\nGender (Male/Female/Other): ").strip(),
            'age': float(input("Age: ").strip()),
            'hypertension': int(input("Hypertension (0=No, 1=Yes): ").strip()),
            'heart_disease': int(input("Heart Disease (0=No, 1=Yes): ").strip()),
            'ever_married': input("Ever Married (Yes/No): ").strip(),
            'work_type': input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ").strip(),
            'Residence_type': input("Residence Type (Urban/Rural): ").strip(),
            'avg_glucose_level': float(input("Average Glucose Level (mg/dL): ").strip()),
            'bmi': float(input("BMI: ").strip()),
            'smoking_status': input("Smoking Status (never smoked/formerly smoked/smokes/Unknown): ").strip()
        }
        
        tester.print_prediction("Custom Patient", user_data)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Please ensure all inputs are in the correct format.")


def main():
    """Main testing function."""
    print("\n" + "="*60)
    print("STROKE PREDICTION SYSTEM - TESTING MODULE")
    print("="*60)
    
    # Load model
    try:
        tester = StrokePredictionTester('models/best_model.pkl')
        print("\n✓ Model loaded successfully!")
    except FileNotFoundError:
        print("\n✗ Error: Model file not found!")
        print("Please train the model first using ml_models.py")
        return
    
    # Menu
    print("\n" + "-"*60)
    print("Select testing mode:")
    print("-"*60)
    print("1. Run batch predictions on predefined test cases")
    print("2. Interactive mode (enter custom patient data)")
    print("3. Run both")
    print("4. Exit")
    print("-"*60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        test_cases = create_test_cases()
        run_batch_predictions(tester, test_cases)
        
    elif choice == '2':
        interactive_mode(tester)
        
    elif choice == '3':
        test_cases = create_test_cases()
        run_batch_predictions(tester, test_cases)
        
        print("\n\n")
        interactive_mode(tester)
        
    elif choice == '4':
        print("\nExiting...")
        return
    
    else:
        print("\n✗ Invalid choice.")
        return
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()