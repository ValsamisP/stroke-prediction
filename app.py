"""
Stroke Prediction User Interface
Interactive web application using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import sklearn
assert sklearn.__version__ == "1.6.1"



class StrokeRiskAssessment:
    """Handle stroke risk prediction and recommendations."""
    
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        """Load the trained model and preprocessing objects."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.label_encoders = self.model_data['label_encoders']
        self.feature_names = self.model_data['feature_names']
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
        else:
            self.feature_importance = None
    
    def preprocess_input(self, user_data: Dict) -> np.ndarray:
        """Preprocess user input data.
        Steps:
        1. One-hot encode categorical variables
        2. Scale numerical features
        3. Ensure correct feature order
        """
        # Create DataFrame
        df = pd.DataFrame([user_data])
        
        # Encode categorical features
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            try:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
            except ValueError as e:
                raise ValueError(f"Invalid value for {col}: {user_data[col]}."
                                 f"Expected one of : {list(self.label_encoders[col].classes_)}")
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        return scaled_data
    
    def predict(self, user_data: Dict) -> Tuple[float, int]:
        """
        Make prediction for user input.
        
        Returns:
            probability: Stroke probability (0-1)
            prediction: Binary prediction (0 or 1)
        """
        processed_data = self.preprocess_input(user_data)
        probability = self.model.predict_proba(processed_data)[0][1]
        prediction = self.model.predict(processed_data)[0]
        
        return probability, prediction
    
    def generate_recommendations(self, user_data: Dict, probability: float) -> List[Dict]:
        """
        Generate personalized recommendations based on feature importance and user data.
        """
        recommendations = []
        
        # Age-based recommendations
        if user_data['age'] > 50:
            recommendations.append({
                'category': 'Age & Monitoring',
                'priority': 'High',
                'recommendation': 'Regular health screenings are crucial. Schedule annual check-ups and monitor cardiovascular health closely.',
                'impact': 'High'
            })
        
        # BMI-based recommendations
        bmi = user_data['bmi']
        if bmi > 30:
            recommendations.append({
                'category': 'Weight Management',
                'priority': 'High',
                'recommendation': f'Your BMI ({bmi:.1f}) indicates obesity. Aim to reduce BMI to 25 or below through diet and exercise. This could reduce stroke risk by 20-30%.',
                'impact': 'High',
                'actionable': 'Consult a healthcare provider for personalized weight loss goals. General target: 0.5-1 kg per week'
            })
        elif bmi > 25:
            recommendations.append({
                'category': 'Weight Management',
                'priority': 'Medium',
                'recommendation': f'Your BMI ({bmi:.1f}) is in the overweight range. Aim for BMI between 18.5-25.',
                'impact': 'Medium',
                'actionable': 'Focus on balanced diet and 150 minutes of moderate exercise per week'
            })
        
        # Glucose level recommendations
        glucose = user_data['avg_glucose_level']
        if glucose > 140:
            recommendations.append({
                'category': 'Blood Sugar Control',
                'priority': 'High',
                'recommendation': f'Your average glucose level ({glucose:.1f} mg/dL) suggests prediabetes or diabetes. This significantly increases stroke risk.',
                'impact': 'High',
                'actionable': 'Consult an endocrinologist, monitor blood sugar daily, reduce sugar/refined carb intake'
            })
        elif glucose > 100:
            recommendations.append({
                'category': 'Blood Sugar Control',
                'priority': 'Medium',
                'recommendation': f'Your glucose level ({glucose:.1f} mg/dL) is elevated. Focus on diet to prevent progression.',
                'impact': 'Medium',
                'actionable': 'Limit sugary foods, increase fiber intake, consider HbA1c testing'
            })
        
        # Hypertension recommendations
        if user_data['hypertension'] == 1:
            recommendations.append({
                'category': 'Blood Pressure Management',
                'priority': 'Critical',
                'recommendation': 'Hypertension is a major stroke risk factor. Ensure it is well-controlled with medication and lifestyle changes.',
                'impact': 'Very High',
                'actionable': 'Monitor BP daily, reduce sodium to <2300mg/day, take prescribed medications, manage stress'
            })
        
        # Heart disease recommendations
        if user_data['heart_disease'] == 1:
            recommendations.append({
                'category': 'Cardiac Health',
                'priority': 'Critical',
                'recommendation': 'Existing heart disease significantly elevates stroke risk. Work closely with a cardiologist.',
                'impact': 'Very High',
                'actionable': 'Regular cardiac monitoring, anticoagulants if prescribed, cardiac rehabilitation program'
            })
        
        # Smoking recommendations
        if user_data['smoking_status'] in ['smokes', 'formerly smoked']:
            if user_data['smoking_status'] == 'smokes':
                recommendations.append({
                    'category': 'Smoking Cessation',
                    'priority': 'Critical',
                    'recommendation': 'Current smoking doubles stroke risk. Quitting is the single most important change you can make.',
                    'impact': 'Very High',
                    'actionable': 'Join smoking cessation program, consider nicotine replacement therapy, set a quit date'
                })
            else:
                recommendations.append({
                    'category': 'Smoking History',
                    'priority': 'Low',
                    'recommendation': 'Great job quitting! Former smokers still have elevated risk for several years.',
                    'impact': 'Medium',
                    'actionable': 'Continue avoiding tobacco, monitor cardiovascular health regularly'
                })
        
        # General lifestyle recommendations
        if probability > 0.1:
            recommendations.append({
                'category': 'Lifestyle & Prevention',
                'priority': 'High',
                'recommendation': 'Adopt a comprehensive stroke prevention lifestyle.',
                'impact': 'High',
                'actionable': 'Mediterranean diet, 30min daily exercise, stress management, 7-8 hours sleep, limit alcohol'
            })
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return recommendations


def create_risk_gauge(probability: float) -> go.Figure:
    """Create a gauge chart for stroke risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Stroke Risk (%)", 'font': {'size': 24}},
        delta={'reference': 5, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': '#d4edda'},
                {'range': [5, 15], 'color': '#fff3cd'},
                {'range': [15, 30], 'color': '#f8d7da'},
                {'range': [30, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'size': 16}
    )
    
    return fig


def create_feature_contribution_chart(user_data: Dict, feature_importance: Dict) -> go.Figure:
    """Create a chart showing feature contributions to risk."""
    # Get top features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    
    features = [f[0] for f in sorted_features]
    importance = [f[1] for f in sorted_features]
    
    # Create readable labels
    feature_labels = {
        'age': f"Age ({user_data['age']} years)",
        'avg_glucose_level': f"Glucose ({user_data['avg_glucose_level']:.0f} mg/dL)",
        'bmi': f"BMI ({user_data['bmi']:.1f})",
        'hypertension': f"Hypertension ({'Yes' if user_data['hypertension'] else 'No'})",
        'heart_disease': f"Heart Disease ({'Yes' if user_data['heart_disease'] else 'No'})",
        'smoking_status': f"Smoking ({user_data['smoking_status']})",
        'ever_married': f"Married ({user_data['ever_married']})",
        'work_type': f"Work ({user_data['work_type']})",
        'gender': f"Gender ({user_data['gender']})",
        'Residence_type': f"Residence ({user_data['Residence_type']})"
    }
    
    labels = [feature_labels.get(f, f) for f in features]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=labels,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Reds',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title="Key Risk Factors (Feature Importance)",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Stroke Risk Prediction",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st.title("🏥 Stroke Risk Prediction & Prevention System")
    st.markdown("""
    This tool uses machine learning to assess your stroke risk and provide personalized recommendations.
    **Note:** This is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Load model
    try:
        assessor = StrokeRiskAssessment('models/best_model.pkl')
    except FileNotFoundError:
        st.error(" Model file not found. Please train the model first using ml_models.py")
        return
    
    # Sidebar for input
    st.sidebar.header("📋 Enter Your Information")
    
    # Collect user input
    user_data = {}
    
    # Demographics
    st.sidebar.subheader("Demographics")
    user_data['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    user_data['age'] = st.sidebar.slider("Age", 0, 100, 50)
    user_data['ever_married'] = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
    user_data['work_type'] = st.sidebar.selectbox(
        "Work Type", 
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    user_data['Residence_type'] = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    
    # Health metrics
    st.sidebar.subheader("Health Metrics")
    user_data['hypertension'] = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    user_data['heart_disease'] = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    user_data['avg_glucose_level'] = st.sidebar.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, 
        max_value=300.0, 
        value=100.0,
        step=0.1
    )
    user_data['bmi'] = st.sidebar.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, 
        max_value=60.0, 
        value=25.0,
        step=0.1
    )
    user_data['smoking_status'] = st.sidebar.selectbox(
        "Smoking Status", 
        ["never smoked", "formerly smoked", "smokes", "Unknown"]
    )
    
    # Predict button
    if st.sidebar.button("🔍 Assess Risk", type="primary"):
        # Make prediction
        probability, prediction = assessor.predict(user_data)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Risk Assessment")
            
            # Risk gauge
            fig_gauge = create_risk_gauge(probability)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk interpretation
            if probability < 0.05:
                risk_level = "Low Risk"
                risk_color = "green"
                risk_message = "Your stroke risk is relatively low. Continue maintaining a healthy lifestyle!"
            elif probability < 0.15:
                risk_level = "Moderate Risk"
                risk_color = "orange"
                risk_message = "Your stroke risk is moderate. Following the recommendations below can help reduce it."
            elif probability < 0.30:
                risk_level = "High Risk"
                risk_color = "red"
                risk_message = "Your stroke risk is high. Please consult with a healthcare provider and follow the recommendations."
            else:
                risk_level = "Very High Risk"
                risk_color = "darkred"
                risk_message = "Your stroke risk is very high. Immediate consultation with a healthcare provider is strongly recommended."
            
            st.markdown(f"### :{risk_color}[{risk_level}]")
            st.info(risk_message)
            
            st.metric(
                "Stroke Probability", 
                f"{probability*100:.2f}%",
                delta=f"{(probability-0.05)*100:.2f}% vs. baseline (5%)"
            )
        
        with col2:
            st.subheader("Risk Factor Analysis")
            
            if assessor.feature_importance:
                fig_features = create_feature_contribution_chart(user_data, assessor.feature_importance)
                st.plotly_chart(fig_features, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        
        # Recommendations
        st.subheader("📌 Personalized Recommendations")
        recommendations = assessor.generate_recommendations(user_data, probability)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_color = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                
                with st.expander(
                    f"{priority_color.get(rec['priority'], '⚪')} {rec['category']} - {rec['priority']} Priority",
                    expanded=(i <= 3)
                ):
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                    if 'actionable' in rec:
                        st.write(f"**Action Steps:** {rec['actionable']}")
                    st.write(f"**Impact on Risk:** {rec['impact']}")
        else:
            st.success("Great! Keep up your healthy lifestyle and continue regular check-ups.")
        
        # Additional info
        st.subheader("📊 Your Health Profile Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Age", f"{user_data['age']} years")
            st.metric("BMI", f"{user_data['bmi']:.1f}")
        
        with col2:
            st.metric("Glucose", f"{user_data['avg_glucose_level']:.0f} mg/dL")
            st.metric("Hypertension", "Yes" if user_data['hypertension'] else "No")
        
        with col3:
            st.metric("Heart Disease", "Yes" if user_data['heart_disease'] else "No")
            st.metric("Smoking Status", user_data['smoking_status'])
        
        # Disclaimer
        st.warning("""
        **⚠️ Important Disclaimer:**
        This prediction tool is for educational and informational purposes only. It should NOT be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other 
        qualified health provider with any questions you may have regarding a medical condition.
        """)


if __name__ == "__main__":
    main()