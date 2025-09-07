# Stroke Prediction using Machine Learning

## Academic Project Overview

This project implements a comprehensive machine learning pipeline for stroke prediction using various classification algorithms. The study addresses the challenge of imbalanced medical datasets where stroke cases represent only 4.9% of the total samples. The project follows a systematic approach with exploratory data analysis, feature engineering, and model selection optimized for medical applications.

## Problem Statement

Stroke is a leading cause of death and disability worldwide. Early prediction of stroke risk can significantly improve patient outcomes through preventive interventions. This project aims to develop an accurate machine learning model to predict stroke occurrence based on patient demographic and medical characteristics, with particular emphasis on minimizing false negatives (missed stroke cases) due to the critical nature of medical diagnosis.

## Dataset Description

The dataset contains 5,110 patient records with the following features:

### Demographic Variables
- **Gender**: Male, Female, Other
- **Age**: Patient age in years
- **Marital Status**: Married/Unmarried
- **Work Type**: Private, Self-employed, Government job, Children, Never worked
- **Residence Type**: Urban, Rural

### Medical Variables
- **Hypertension**: Binary indicator (0/1)
- **Heart Disease**: Binary indicator (0/1)
- **Average Glucose Level**: Blood glucose concentration (mg/dL)
- **BMI**: Body Mass Index
- **Smoking Status**: Never smoked, Formerly smoked, Smokes, Unknown

### Target Variable
- **Stroke**: Binary outcome (0 = No stroke, 1 = Stroke)

## Methodology

### 1. Exploratory Data Analysis (EDA)
The analysis is conducted in `Notebooks/EDA.ipynb` and includes:

- **Data Quality Assessment**: Missing value analysis, outlier detection, data type verification
- **Descriptive Statistics**: Distribution analysis for all variables
- **Univariate Analysis**: Individual variable examination with appropriate visualizations
- **Bivariate Analysis**: Relationship analysis between variables and stroke outcome
- **Multivariate Analysis**: Complex interaction patterns between multiple risk factors
- **Data Transformation**: Feature engineering and preparation for machine learning

### 2. Feature Engineering and Model Selection
The modeling process is conducted in `Notebooks/Feature.ipynb` and includes:

#### Data Preprocessing Pipeline
- **Missing Value Imputation**: Median imputation for BMI (201 missing values)
- **Outlier Removal**: IQR-based outlier detection and removal
- **Categorical Encoding**: OneHotEncoder for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Power Transformation**: Yeo-Johnson transformation for skewed variables
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)

#### Model Selection Strategy
The following machine learning algorithms were systematically evaluated:

1. **K-Nearest Neighbors**: Selected as the optimal model
2. **Random Forest**: Ensemble method with balanced class weights
3. **Decision Tree**: Interpretable single-tree model
4. **Gradient Boosting**: Sequential ensemble learning
5. **Support Vector Machine**: Kernel-based classification
6. **Logistic Regression**: Linear baseline model
7. **Naive Bayes**: Probabilistic classifier

### 3. Model Evaluation
- **Primary Metric**: Recall (Sensitivity) - Critical for medical applications
- **Secondary Metrics**: F1-Score, Precision, AUC-ROC, Accuracy
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Optimization**: RandomizedSearchCV for optimal parameters

## Project Structure

```
stroke_prediction/
├── data/
│   ├── stroke_data.csv              # Original dataset
│   └── stroke_data_clean.csv        # Processed dataset
├── doc/
│   └── Project_Description_SPOC_Python_ML.docx.pdf
├── Notebooks/
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   └── Feature.ipynb                # Feature Engineering & Model Selection
├── main.py                          # Main application interface
├── README.md                        # Project documentation
└── pyproject.toml                   # Project configuration
```

## Installation and Setup

### Prerequisites
- Python 3.13+
- pip or uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd stroke_prediction

# Install dependencies
uv sync
# or
pip install -e .
```

### Dependencies
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `imbalanced-learn>=0.11.0` - Class balancing techniques
- `xgboost>=2.0.0` - Extreme gradient boosting
- `pandas>=2.3.2` - Data manipulation
- `numpy>=2.3.2` - Numerical computations
- `matplotlib>=3.10.6` - Data visualization
- `seaborn>=0.13.2` - Statistical plotting
- `joblib>=1.3.0` - Model persistence

## Usage

### Running the Analysis

#### 1. Exploratory Data Analysis
```bash
# Open and run the EDA notebook
jupyter notebook Notebooks/EDA.ipynb
```

#### 2. Feature Engineering and Model Selection
```bash
# Open and run the Feature Engineering notebook
jupyter notebook Notebooks/Feature.ipynb
```

#### 3. Interactive Prediction Application
```bash
python main.py
```

The main application provides a user-friendly interface with the following options:
- **Interactive Prediction**: Enter patient data manually for stroke risk assessment
- **Model Information**: View details about the trained model and its performance
- **Exit**: Close the application

### Interactive Prediction Interface

The main application (`main.py`) provides an interface to test the trained model:

#### Input Requirements:
- **Demographic**: Gender, Age, Marital Status, Work Type, Residence Type
- **Medical**: Hypertension, Heart Disease, Average Glucose Level, BMI, Smoking Status

#### Output Information:
- **Risk Level**: Low (🟢), Moderate (🟡), High (🟠), Very High (🔴)
- **Probability Analysis**: Stroke and No-Stroke probabilities
- **Confidence Score**: Model prediction confidence
- **Medical Recommendations**: Tailored advice based on risk level

### Programmatic Usage

```python
from main import StrokePredictionApp

app = StrokePredictionApp()
app.load_trained_model()

# Patient data
patient_data = {
    'gender': 'Male',
    'age': 65,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 150.0,
    'bmi': 28.5,
    'smoking_status': 'formerly smoked'
}

result = app.predict_single_patient(patient_data)
print(f"Stroke probability: {result['probability_stroke']:.2%}")
print(f"Risk level: {result['risk_level']}")
```

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **K-Nearest Neighbors** | **90.3%** | **84.6%** | **98.6%** | **91.1%** | **0.95+** |
| Random Forest | 94.4% | 92.5% | 96.6% | 94.5% | 0.98+ |
| Decision Tree | 89.9% | 88.2% | 92.0% | 90.1% | 0.92+ |
| Gradient Boosting | 88.6% | 86.2% | 91.8% | 88.9% | 0.90+ |
| Support Vector Machine | 84.7% | 79.9% | 92.7% | 85.8% | 0.88+ |
| Logistic Regression | 79.2% | 75.8% | 85.5% | 80.4% | 0.85+ |
| Naive Bayes | 64.3% | 58.9% | 94.6% | 72.6% | 0.80+ |

### Model Selection Rationale

**K-Nearest Neighbors was selected as the optimal model** for the following reasons:

1. **Highest Recall (98.6%)**: Critical for medical applications where missing a true stroke case could have severe consequences
2. **Medical Priority**: In stroke prediction, false negatives are more dangerous than false positives
3. **Robust Performance**: Consistent results across different patient populations
4. **Interpretability**: Easy to understand and explain to medical professionals

### Key Findings

1. **K-Nearest Neighbors** achieved the highest recall (98.6%) making it optimal for medical applications
2. **Age** was identified as the most significant risk factor
3. **Hypertension** and **heart disease** showed strong predictive power
4. **Class balancing** with SMOTE significantly improved model performance on minority class
5. **Feature engineering** enhanced model performance through proper encoding and transformation

## Risk Assessment

The system provides risk levels based on stroke probability:

- **Low Risk** (< 20%): Continue healthy lifestyle, routine monitoring
- **Moderate Risk** (20-50%): Regular medical monitoring recommended
- **High Risk** (50-80%): Urgent medical consultation advised
- **Very High Risk** (> 80%): Immediate medical attention required

## Application Features

### Main Application (`main.py`)

The main application provides a stroke prediction system with the following capabilities:

#### 1. Interactive Prediction Mode
- **Step-by-step data collection** with input validation
- **Real-time risk assessment** using the optimized KNN model
- **Comprehensive result display** with medical recommendations
- **Error handling** for invalid inputs

#### 2. Model Information Display
- **Algorithm details**: K-Nearest Neighbors with optimized hyperparameters
- **Feature information**: Number of selected features used for prediction
- **Performance metrics**: Model optimization focused on medical applications
- **Use case description**: Stroke risk assessment system

#### 3. Robust Error Handling
- **File validation**: Checks for required model files before startup
- **Input validation**: Ensures all patient data is within valid ranges
- **Graceful error recovery**: Clear error messages and recovery options
- **Keyboard interrupt handling**: Clean exit with Ctrl+C

### Model Deployment

The application automatically loads the following files:
- `models/best_model.pkl` - Optimized KNN model
- `models/scaler.pkl` - Feature scaling object
- `models/selected_features.pkl` - Selected feature list
- `models/model_metadata.pkl` - Model performance metadata


## Technical Notes

### Data Processing
- **Missing Values**: 201 missing BMI values imputed with median
- **Outliers**: IQR-based outlier removal for improved model stability
- **Class Imbalance**: SMOTE applied to balance stroke/no-stroke classes
- **Feature Scaling**: StandardScaler applied to numerical features

### Model Optimization
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameter selection
- **Cross-Validation**: 5-fold stratified CV for robust performance estimation
- **Feature Selection**: Statistical and tree-based feature importance analysis

## References

1. World Health Organization. (2021). Stroke, Cerebrovascular accident.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority oversampling technique.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
