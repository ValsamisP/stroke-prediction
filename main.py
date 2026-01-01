"""
Main execution script for the complete Stroke Prediction project.
Run this file to execute the entire pipeline from EDA to model training.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for outputs."""
    directories = ['plots', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")

def run_eda(dataset_path):
    """Run exploratory data analysis."""
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    try:
        from visualization import run_complete_eda
        run_complete_eda(dataset_path, save_dir='plots')
        print("\n✓ EDA completed successfully!")
        print("✓ Visualizations saved in plots/ directory")
    except Exception as e:
        print(f"✗ Error during EDA: {str(e)}")
        return False
    
    return True

def run_model_training(dataset_path):
    """Train and evaluate all ML models."""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("="*70)
    
    try:
        from ml_models import run_complete_pipeline
        predictor, results = run_complete_pipeline(dataset_path, save_dir='models')
        
        print("\n✓ Model training completed successfully!")
        print("✓ Best model saved in models/ directory")
        print("✓ Model comparison results saved in models/model_comparison.csv")
        
        return predictor, results
    except Exception as e:
        print(f"✗ Error during model training: {str(e)}")
        return None, None

def launch_app():
    """Launch the Streamlit application."""
    print("\n" + "="*70)
    print("STEP 3: LAUNCHING WEB APPLICATION")
    print("="*70)
    
    print("\nTo launch the web application, run:")
    print("\n    streamlit run app.py\n")
    print("The application will open in your default web browser.")
    print("You can then enter patient data and get stroke risk predictions!")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STROKE PREDICTION & PREVENTION SYSTEM")
    print("Complete Machine Learning Pipeline")
    print("="*70)
    
    # Configuration
    DATASET_PATH = 'healthcaredatasetstrokedata.csv'
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ Error: Dataset not found at {DATASET_PATH}")
        print("Please ensure the dataset file is in the current directory.")
        sys.exit(1)
    
    print(f"\n✓ Dataset found: {DATASET_PATH}")
    
    # Create directories
    print("\nCreating output directories...")
    create_directories()
    
    # Ask user which steps to run
    print("\n" + "-"*70)
    print("What would you like to do?")
    print("-"*70)
    print("1. Run complete pipeline (EDA + Model Training)")
    print("2. Run EDA only")
    print("3. Run Model Training only")
    print("4. Launch Web Application only")
    print("5. Exit")
    print("-"*70)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Run complete pipeline
        print("\nRunning complete pipeline...")
        
        # EDA
        if not run_eda(DATASET_PATH):
            print("\n✗ Pipeline stopped due to EDA error")
            sys.exit(1)
        
        # Model Training
        predictor, results = run_model_training(DATASET_PATH)
        if predictor is None:
            print("\n✗ Pipeline stopped due to training error")
            sys.exit(1)
        
        # Instructions for web app
        launch_app()
        
    elif choice == '2':
        # EDA only
        run_eda(DATASET_PATH)
        
    elif choice == '3':
        # Model Training only
        run_model_training(DATASET_PATH)
        
    elif choice == '4':
        # Launch web app
        if not os.path.exists('models/best_model.pkl'):
            print("\n✗ Error: No trained model found!")
            print("Please run model training first (option 1 or 3)")
            sys.exit(1)
        
        import subprocess
        print("\nLaunching Streamlit application...")
        subprocess.run(['streamlit', 'run', 'app.py'])
        
    elif choice == '5':
        print("\nExiting...")
        sys.exit(0)
        
    else:
        print("\n✗ Invalid choice. Please run again and select 1-5.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("- Review the generated visualizations in plots/")
    print("- Check model performance in models/model_comparison.csv")
    print("- Launch the web app with: streamlit run app.py")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()