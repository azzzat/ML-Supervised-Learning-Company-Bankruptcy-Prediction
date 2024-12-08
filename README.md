## 📊 Company Bankruptcy Prediction

This repository contains an end-to-end machine learning pipeline to predict whether a company is likely to go bankrupt based on its financial data. The project employs various supervised learning techniques and advanced preprocessing methods to achieve high predictive accuracy.
The data were collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

## 📁 Project Structure

    data/: Dataset used for training and testing the models.
    notebooks/: Jupyter notebooks detailing the exploratory data analysis (EDA), 
    preprocessing, and model evaluation.
    src/: Python scripts for feature engineering, model training, and evaluation.
    results/: Visualizations, reports, and saved models.

## 🧠 Problem Statement

Predict the likelihood of company bankruptcy based on financial indicators. The dataset is highly imbalanced, with far fewer bankrupt companies compared to non-bankrupt ones. The challenge involves handling class imbalance, feature selection, and achieving high accuracy with interpretable results.

## ⚙️ Key Features

    Exploratory Data Analysis (EDA):
        Class distribution analysis.
        Feature correlation analysis.

    Data Preprocessing:
        Removal of constant, duplicate, and highly correlated features.
        Class balancing using SMOTE.
        Dimensionality reduction via PCA.

    Model Training:
        Supervised learning models:
            Logistic Regression
            Decision Tree
            K Neighbors Classifier
            Random Forest Classifier
            Stacking Classifier
            CatBoost
        Evaluation with accuracy, classification reports, and confusion matrices.

    Ensemble Learning:
        Stacking Classifier achieved the best performance with 98.07% accuracy.

## 🔍 Results

Model	Accuracy
Logistic Regression	88.64%
Decision Tree Classifier	92.88%
K Neighbors Classifier	93.79%
Random Forest Classifier	96.78%
Stacking Classifier	98.07%
CatBoost	97.35%

    The Stacking Classifier demonstrated the best predictive power 
    by combining multiple base models effectively.

## 🛠️ Tools & Libraries

    Python: Core programming language for the project.
    Pandas, NumPy: Data manipulation and preprocessing.
    Matplotlib, Seaborn: Visualization.
    Scikit-learn: Model implementation and evaluation.
    CatBoost: Gradient boosting framework.
