# Titanic - Machine Learning from Disaster

This repository contains my solution to the Titanic Kaggle competition, where the goal is to predict the survival of passengers on the Titanic. My best result achieved a score of **0.78468** using a Logistic Regression model.

https://www.kaggle.com/competitions/3136/images/header

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Approach](#approach)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Best Model](#best-model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction
The Titanic competition is a popular beginner challenge on Kaggle, where participants build models to predict whether a passenger survived the Titanic disaster based on features like age, sex, and ticket class.

## Data
- **Training Set**: 891 examples with 11 features.
- **Test Set**: 418 examples for prediction.

## Approach
I explored various models and preprocessing techniques to improve prediction accuracy. The focus was on feature engineering, model tuning, and ensemble methods.

## Preprocessing
- **Imputation**: Missing values were handled using the `SimpleImputer` from scikit-learn.
- **Encoding**: Categorical features were transformed using `OneHotEncoder`.
- **Scaling**: `StandardScaler` was applied to numerical features.
- **Feature Selection**: The most relevant features were selected for modeling.

## Models Used
1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Classifier (SVC)**
4. **K-Nearest Neighbors (KNN)**
5. **XGBoost**
6. **Voting Classifier**: An ensemble of multiple models.

## Best Model
The best-performing model was **Logistic Regression**, which achieved a Kaggle competition score of **0.78468**.

## Results
The Logistic Regression model outperformed other models with minimal tuning, demonstrating its effectiveness for this task.

## Installation
To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow
```

## Usage
1. Clone this repository.
2. Run the Jupyter notebook `main.ipynb`.
3. Follow the instructions in the notebook to preprocess the data, train the models, and generate predictions.

## Conclusion
This project provided an insightful experience into data preprocessing, feature engineering, and model selection. The Logistic Regression model's strong performance highlights its potential for similar binary classification problems.
