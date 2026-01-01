"""
Visualization functions for Stroke Prediction Dataset
Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial preprocessing of stroke data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Handle BMI missing values (N/A as string)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['stroke'].value_counts()}")
    print(f"\nClass imbalance ratio: {df['stroke'].value_counts()[0] / df['stroke'].value_counts()[1]:.2f}:1")
    
    return df


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot the distribution of stroke cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    stroke_counts = df['stroke'].value_counts()
    axes[0].bar(['No Stroke', 'Stroke'], stroke_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Stroke Cases Distribution')
    for i, v in enumerate(stroke_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Percentage plot
    stroke_pct = df['stroke'].value_counts(normalize=True) * 100
    axes[1].pie(stroke_pct.values, labels=['No Stroke', 'Stroke'], 
                autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Stroke Cases Percentage')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot distributions of numerical features by stroke status."""
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        # Distribution plot
        ax1 = axes[idx]
        for stroke_val in [0, 1]:
            data = df[df['stroke'] == stroke_val][col].dropna()
            ax1.hist(data, alpha=0.6, bins=30, 
                    label=f'{"No Stroke" if stroke_val == 0 else "Stroke"}',
                    color='#2ecc71' if stroke_val == 0 else '#e74c3c')
        ax1.set_xlabel(col.replace('_', ' ').title())
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{col.replace("_", " ").title()} Distribution')
        ax1.legend()
        
        # Box plot
        ax2 = axes[idx + 3]
        df_clean = df.dropna(subset=[col])
        bp = ax2.boxplot([df_clean[df_clean['stroke'] == 0][col],
                          df_clean[df_clean['stroke'] == 1][col]],
                         labels=['No Stroke', 'Stroke'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax2.set_ylabel(col.replace('_', ' ').title())
        ax2.set_title(f'{col.replace("_", " ").title()} by Stroke Status')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot distributions of categorical features by stroke status."""
    categorical_cols = ['gender', 'hypertension', 'heart_disease', 
                       'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Create crosstab
        ct = pd.crosstab(df[col], df['stroke'], normalize='index') * 100
        
        # Plot
        ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Stroke Rate by {col.replace("_", " ").title()}')
        ax.legend(['No Stroke', 'Stroke'])
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    # Hide extra subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot correlation matrix of numerical features."""
    # Prepare data for correlation
    df_corr = df.copy()
    
    # Encode binary categorical variables
    df_corr['gender_encoded'] = df_corr['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    df_corr['ever_married_encoded'] = df_corr['ever_married'].map({'Yes': 1, 'No': 0})
    df_corr['Residence_type_encoded'] = df_corr['Residence_type'].map({'Urban': 1, 'Rural': 0})
    
    # Select numerical and encoded columns
    corr_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                 'gender_encoded', 'ever_married_encoded', 'Residence_type_encoded', 'stroke']
    
    correlation_matrix = df_corr[corr_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlations with stroke
    print("\nCorrelations with Stroke (sorted by absolute value):")
    stroke_corr = correlation_matrix['stroke'].drop('stroke').abs().sort_values(ascending=False)
    for feature, corr in stroke_corr.items():
        print(f"{feature}: {correlation_matrix.loc[feature, 'stroke']:.3f}")


def plot_age_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """Detailed age analysis with stroke risk."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Age bins
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    # Stroke rate by age group
    age_stroke = df.groupby('age_group')['stroke'].agg(['sum', 'count', 'mean'])
    axes[0].bar(range(len(age_stroke)), age_stroke['mean'] * 100, color='#3498db')
    axes[0].set_xticks(range(len(age_stroke)))
    axes[0].set_xticklabels(age_stroke.index, rotation=45)
    axes[0].set_ylabel('Stroke Rate (%)')
    axes[0].set_title('Stroke Rate by Age Group')
    axes[0].grid(True, alpha=0.3)
    
    # Add labels
    for i, v in enumerate(age_stroke['mean'] * 100):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Age distribution by stroke status
    axes[1].hist([df[df['stroke'] == 0]['age'], df[df['stroke'] == 1]['age']], 
                bins=20, alpha=0.7, label=['No Stroke', 'Stroke'],
                color=['#2ecc71', '#e74c3c'])
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Age Distribution by Stroke Status')
    axes[1].legend()
    
    # Cumulative stroke risk by age
    age_sorted = df.sort_values('age')
    age_sorted['cumulative_stroke_rate'] = (
        age_sorted['stroke'].expanding().mean() * 100
    )
    axes[2].plot(age_sorted['age'], age_sorted['cumulative_stroke_rate'], 
                color='#e74c3c', linewidth=2)
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Cumulative Stroke Rate (%)')
    axes[2].set_title('Cumulative Stroke Risk by Age')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""
    summary_stats = []
    
    # Numerical features
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    
    for col in numerical_cols:
        for stroke_val in [0, 1]:
            data = df[df['stroke'] == stroke_val][col].dropna()
            summary_stats.append({
                'Feature': col,
                'Stroke': 'Yes' if stroke_val == 1 else 'No',
                'Mean': data.mean(),
                'Median': data.median(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max()
            })
    
    summary_df = pd.DataFrame(summary_stats)
    print("\nSummary Statistics by Stroke Status:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def run_complete_eda(filepath: str, save_dir: Optional[str] = None):
    """
    Run complete exploratory data analysis.
    
    Args:
        filepath: Path to the CSV file
        save_dir: Directory to save plots (optional)
    """
    print("="*60)
    print("STROKE PREDICTION DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_and_preprocess_data(filepath)
    
    # Generate all visualizations
    print("\n1. Plotting class distribution...")
    plot_class_distribution(df, f"{save_dir}/class_distribution.png" if save_dir else None)
    
    print("\n2. Plotting numerical distributions...")
    plot_numerical_distributions(df, f"{save_dir}/numerical_distributions.png" if save_dir else None)
    
    print("\n3. Plotting categorical distributions...")
    plot_categorical_distributions(df, f"{save_dir}/categorical_distributions.png" if save_dir else None)
    
    print("\n4. Plotting correlation matrix...")
    plot_correlation_matrix(df, f"{save_dir}/correlation_matrix.png" if save_dir else None)
    
    print("\n5. Plotting age analysis...")
    plot_age_analysis(df, f"{save_dir}/age_analysis.png" if save_dir else None)
    
    print("\n6. Generating summary statistics...")
    generate_summary_statistics(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    run_complete_eda('healthcaredatasetstrokedata.csv', save_dir='plots')