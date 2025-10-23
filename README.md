# Predictive Maintenance in Industrial Settings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MSc Data Science Advanced Practice Project** | Teesside University  
> A comprehensive machine learning solution for predicting equipment failures in manufacturing environments

## 🎯 Project Overview

This project implements multiple machine learning algorithms to predict machine failures in industrial settings, achieving up to **99.80% accuracy**. The system identifies five distinct failure types and provides explainable predictions using LIME (Local Interpretable Model-agnostic Explanations).

### Key Achievements
- ✅ **99.80% Test Accuracy** with Support Vector Machines
- ✅ **Multi-class Classification** of 5 failure types
- ✅ **Advanced Data Balancing** using SMOTE technique
- ✅ **Comprehensive EDA** with detailed visualizations
- ✅ **Model Interpretability** using LIME explainability

---

## 📊 Dataset Description

**Source:** [Kaggle - Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

The dataset contains **10,000 observations** with 14 features representing synthetic industrial sensor data:

### Features
| Feature | Description | Type |
|---------|-------------|------|
| **Type** | Product quality (L/M/H: 60%/30%/10%) | Categorical |
| **Air Temperature** | Ambient temperature (K) | Continuous |
| **Process Temperature** | Operating temperature (K) | Continuous |
| **Rotational Speed** | RPM from 2860W power | Continuous |
| **Torque** | Torque in Nm | Continuous |
| **Tool Wear** | Tool usage time (minutes) | Continuous |
| **Target** | Machine failure indicator (0/1) | Binary |
| **Failure Type** | Specific failure mode | Categorical |

### Failure Types
1. **Tool Wear Failure (TWF):** Tool replacement at 200-240 mins
2. **Heat Dissipation Failure (HDF):** Temp difference <8.6K AND speed <1380 RPM
3. **Power Failure (PWF):** Power <3500W OR >9000W
4. **Overstrain Failure (OSF):** Tool wear × torque >11,000 minNm
5. **Random Failures (RNF):** 0.1% random failure probability

---

## 🛠️ Technical Stack

### Core Libraries
```python
# Data Processing
pandas==1.5.3
numpy==1.23.5

# Machine Learning
scikit-learn==1.2.2
imbalanced-learn==0.10.1
tensorflow==2.12.0
xgboost==1.7.5

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Model Interpretation
lime==0.2.0.1

# Utilities
category-encoders==2.6.0
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- **Data Quality Checks:** No missing values, no duplicates
- **Outlier Detection:** Identified legitimate outliers in Rotational Speed and Torque
- **Feature Engineering:** Created Temperature Difference feature
- **Correlation Analysis:** Heatmap revealing key relationships

### 2. Data Preprocessing

#### Handling Class Imbalance
- **Original Distribution:** 96.69% No Failure, 3.31% Failure
- **SMOTE Application:** Balanced to 80% No Failure, 20% Failure
- **Result:** 20.88% increase in observations (9,973 → 12,055)

#### Feature Scaling & Encoding
- **StandardScaler** for numerical features
- **Label Encoding** for categorical features
- **PCA Analysis:** 3 components explain 85.49% variance

### 3. Model Training

Trained and compared **9 machine learning algorithms** with GridSearchCV hyperparameter tuning:

| Model | Training Accuracy | Test Accuracy | Key Strength |
|-------|------------------|---------------|--------------|
| **Support Vector Machine** | 99.91% | **99.80%** | Best overall performance |
| Gradient Boosting | 100.00% | 99.70% | Strong ensemble |
| Random Forest | 100.00% | 99.65% | Robust predictions |
| Decision Tree | 100.00% | 99.60% | Fast training |
| Logistic Regression | 99.82% | 99.60% | Interpretable |
| Extra Trees | 100.00% | 99.50% | Low variance |
| AdaBoost | 99.76% | 99.50% | Adaptive learning |
| Naïve Bayes | 99.77% | 99.40% | Fast inference |
| k-Nearest Neighbors | 100.00% | 97.29% | Simple approach |

### 4. Model Evaluation

**Best Model: Support Vector Machine (RBF Kernel)**

#### Classification Report
```
                          precision    recall  f1-score   support
No Failure                    1.00      1.00      1.00      1930
Power Failure                 0.96      0.96      0.96        27
Overstrain Failure            0.89      1.00      0.94         8
Heat Dissipation Failure      0.86      0.86      0.86        14
Tool Wear Failure             1.00      0.94      0.97        16

Accuracy                                          1.00      1995
Macro Avg                     0.94      0.95      0.95      1995
Weighted Avg                  1.00      1.00      1.00      1995
```

---

## 📁 Project Structure

```
Predictive-Maintenance-Using-ML/
│
├── AP.ipynb                          # Main analysis notebook
├── data.csv                          # Dataset
├── README.md                         # This file
├── requirements.txt                  # Dependencies
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # EDA details
│
├── models/
│   ├── svm_model.pkl                 # Trained SVM model
│   └── scaler.pkl                    # Feature scaler
│
└── visualizations/
    ├── confusion_matrix.png
    ├── pca_3d.png
    └── correlation_heatmap.png
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab

### Installation

```bash
# Clone the repository
git clone https://github.com/UsamaMasood12/Predictive-Maintenance-Using-ML.git
cd Predictive-Maintenance-Using-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook AP.ipynb
```

### Quick Start

```python
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your data
data = pd.read_csv('your_machine_data.csv')

# Preprocess and predict
predictions = model.predict(data)
```

---

## 📈 Key Insights

### From EDA
1. **Temperature Relationships:** Strong correlation (0.87) between Air and Process temperature
2. **Power Dynamics:** Inverse relationship between Torque and Rotational Speed
3. **Tool Wear Impact:** Most significant predictor for failure (0.32 correlation with target)
4. **Quality Variance:** Low-quality machines (Type L) show higher failure rates

### From PCA
- **PC1 (38.11%):** Temperature-related features
- **PC2 (30.84%):** Power-related (Torque × Rotational Speed)
- **PC3 (16.55%):** Tool Wear

### From LIME Explainability
- **Torque >46.70** is the strongest predictor for Class 1 (failure)
- **Tool Wear** shows clear separation between failure types
- **Target ≤0.00** strongly indicates no failure

---

## 🔧 Model Deployment

### Save Model
```python
import pickle

# Save the trained SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(grid_svc.best_estimator_, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)
```

### Load and Use
```python
# Load model and scaler
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
scaled_features = scaler.transform(new_data)
predictions = model.predict(scaled_features)
probabilities = model.predict_proba(scaled_features)
```

---

## 📊 Results Visualization

### Confusion Matrix
Perfect classification with minimal false positives across all failure types.

### 3D PCA Visualization
Clear separation of failure types in reduced dimensional space, validating model performance.

### Feature Importance
Tool Wear, Torque, and Rotational Speed emerge as the most critical predictors.

---

## 🎓 Academic Context

**Institution:** Teesside University  
**Program:** MSc Data Science with Advanced Practice  
**Supervisor:** [Supervisor Name]  
**Date:** April 2025

This project demonstrates:
- Advanced machine learning techniques
- Real-world problem-solving in industrial IoT
- Data science best practices
- Model interpretability and explainability

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Usama Masood**
- 📧 Email: usamamasood531@gmail.com
- 💼 LinkedIn: [linkedin.com/in/usama-masood-b4a35014b](https://www.linkedin.com/in/usama-masood-b4a35014b)
- 🐙 GitHub: [@UsamaMasood12](https://github.com/UsamaMasood12)

---

## 🙏 Acknowledgments

- Kaggle for providing the Machine Predictive Maintenance dataset
- Teesside University for academic support
- The open-source community for excellent ML libraries

---

## 📚 References

1. Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Dataset." NASA Ames Prognostics Data Repository.
2. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research.
3. Ribeiro, M. T., et al. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD.

---

**⭐ If you find this project helpful, please consider giving it a star!**
