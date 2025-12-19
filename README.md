# Car-Sales-Predictive-model

🚗 Car Price and Accident Risk Prediction Using Machine Learning
📌 Project Overview

This project focuses on predicting car selling prices and accident risk using Machine Learning techniques.
A real-world automotive dataset containing 50,000 car records was analyzed, preprocessed, visualized, and modeled to derive meaningful insights.

The project demonstrates an end-to-end predictive analytics workflow, including:

Data preprocessing

Exploratory Data Analysis (EDA)

Feature scaling and encoding

Regression and classification modeling

Model evaluation using appropriate metrics

🎯 Objectives

Predict Car Price using regression models

Predict Accident History (Yes/No) using classification models

Compare multiple machine learning models

Understand key factors influencing price and accident risk

📂 Dataset Information

Source: Kaggle

Link: https://www.kaggle.com/datasets/mahdimashayekhi/used-car-price

Records: 50,000

Features: 25

Key Attributes:

Brand, Model, Year, Car Age

Mileage, Engine Size, Horsepower, Torque

Fuel Type, Transmission, Drive Type

Accident History, Fuel Efficiency

City, Condition, Price

🧹 Data Preprocessing

The following preprocessing steps were applied:

Handled missing values

Numeric columns → median

Categorical columns → mode

Converted AccidentHistory from Yes/No → 1/0

Encoded categorical features using Label Encoding

Applied StandardScaler for feature normalization

Handled class imbalance using SMOTE

📊 Exploratory Data Analysis (EDA)

EDA was performed using Matplotlib and Seaborn to understand data patterns:

Price and mileage distribution

Fuel type and transmission count

Brand popularity

Condition vs price (boxplot)

Correlation heatmap

Scatter plots and pairplots

Key Insights:

Price decreases as mileage and age increase

Engine size and horsepower positively influence price

Accident history impacts resale value

🤖 Machine Learning Models Used
🔹 Regression Models (Price Prediction)
Model	Purpose
Linear Regression	Baseline price prediction
Polynomial Regression	Capture non-linear price patterns

Evaluation Metric: Mean Squared Error (MSE)

🔹 Classification Models (Accident History Prediction)
Model	Purpose
Logistic Regression	Binary classification
Decision Tree	Rule-based classification
K-Nearest Neighbors (KNN)	Distance-based classification
Naive Bayes	Probabilistic classification

Evaluation Metrics:

Accuracy

Confusion Matrix

Precision, Recall, F1-score

Graphical classification reports were also generated using heatmaps.

📈 Results Summary

Polynomial Regression achieved the lowest MSE for price prediction

Decision Tree Classifier performed best for accident history prediction

Feature scaling improved model performance

SMOTE helped handle class imbalance effectively

🧠 Why Two Different Predictions?

Price Prediction:

Continuous numeric value → Regression models

Accident History Prediction:

Binary categorical output (0/1) → Classification models

🛠 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn (SMOTE)

📁 Project Structure
├── car_price_dataset.csv
├── carsalesmodels.py
├── README.md

🔮 Future Scope

Add advanced ensemble models

Hyperparameter tuning

Feature selection techniques

Deploy model using Flask or Streamlit

Integrate real-time car market data

👩‍🏫 Academic Use

This project was developed as part of a Machine Learning / Predictive Analytics academic project and is intended for learning and educational purposes.
