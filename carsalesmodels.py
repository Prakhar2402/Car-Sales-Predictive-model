# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE




# 2. LOAD DATASET
df = pd.read_csv("C:\\Users\\asus\\Downloads\\car_price_dataset.csv")
print("\nDataset Loaded Successfully\n")



# 3. EDA
print(df.head())
print(df.info())
print(df.describe(include="all"))

print("\nNull Values:\n", df.isnull().sum())
print("\nUnique Values:\n", df.nunique())



# 4. HANDLE NULL VALUES
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\nNulls After Cleaning:\n", df.isnull().sum())

# Convert Accident History Yes/No → 1/0
print(df['AccidentHistory'].unique())
df['AccidentHistory'] = df['AccidentHistory'].map({'Yes': 1, 'No': 0})



# 5. VISUALIZATIONS

plt.figure(figsize=(6,4))
sns.histplot(df["Price($)"], kde=True, color="purple")
plt.title("Price Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Mileage(km)"], kde=True, color="green")
plt.title("Mileage Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="FuelType", data=df, palette="viridis")
plt.title("Fuel Type Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Transmission", data=df, palette="magma")
plt.title("Transmission Type Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,4))
df["Brand"].value_counts().head(10).plot(kind='bar', color="teal")
plt.title("Top 10 Most Common Car Brands")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x="Condition", y="Price($)", data=df, palette="coolwarm")
plt.title("Price Variation by Car Condition")
plt.show()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Mileage(km)"], y=df["Price($)"], color="red")
plt.title("Mileage vs Price")
plt.show()

sns.pairplot(df[["Price($)", "Mileage(km)", "EngineSize(L)", "Horsepower"]], palette="husl")
plt.show()



# 6. LABEL ENCODING FOR OTHER CATEGORICALS

le = LabelEncoder()
for col in cat_cols:
    if col != "AccidentHistory":
        df[col] = le.fit_transform(df[col])



# 7. NORMALIZATION

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# 8. TRAIN–TEST SPLIT for PRICE PREDICTION

X = df.drop("Price($)", axis=1)
y = df["Price($)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# MODEL 1 — LINEAR REGRESSION
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
print("\nLinear Regression MSE:", mse_lin)


# MODEL 2 — POLYNOMIAL REGRESSION
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_reg = LinearRegression()
poly_reg.fit(X_train_p, y_train_p)

y_pred_poly = poly_reg.predict(X_test_p)

mse_poly = mean_squared_error(y_test_p, y_pred_poly)
print("Polynomial Regression MSE:", mse_poly)



# 9. ACCIDENT HISTORY CLASSIFICATION

Xc = df.drop("AccidentHistory", axis=1)
yc = df["AccidentHistory"]

# Apply SMOTE for class imbalance
sm = SMOTE()
Xc_resampled, yc_resampled = sm.fit_resample(Xc, yc)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_resampled, yc_resampled, test_size=0.2, random_state=42)



# MODEL 3 — LOGISTIC REGRESSION
log = LogisticRegression(max_iter=2000)
log.fit(Xc_train, yc_train)
pred_log = log.predict(Xc_test)
print("\nLogistic Regression Accuracy:", accuracy_score(yc_test, pred_log))



# MODEL 4 — DECISION TREE
dt = DecisionTreeClassifier()
dt.fit(Xc_train, yc_train)
pred_dt = dt.predict(Xc_test)
print("Decision Tree Accuracy:", accuracy_score(yc_test, pred_dt))



# MODEL 5 — KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xc_train, yc_train)
pred_knn = knn.predict(Xc_test)
print("KNN Accuracy:", accuracy_score(yc_test, pred_knn))



# MODEL 6 — NAIVE BAYES
nb = GaussianNB()
nb.fit(Xc_train, yc_train)
pred_nb = nb.predict(Xc_test)
print("Naive Bayes Accuracy:", accuracy_score(yc_test, pred_nb))



# CONFUSION MATRICES & CLASSIFICATION REPORT HEATMAP

models = {
    "Logistic Regression": pred_log,
    "Decision Tree": pred_dt,
    "KNN": pred_knn,
    "Naive Bayes": pred_nb
}

for name, preds in models.items():
    print("\nModel:", name)
    print("Confusion Matrix:\n", confusion_matrix(yc_test, preds))
    print("Classification Report:\n", classification_report(yc_test, preds))

    # Graphical Classification Report
    report = classification_report(yc_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).T

    plt.figure(figsize=(6,4))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="crest")
    plt.title(f"{name} - Classification Report Heatmap")
    plt.show()




"""
📌 FINAL CONCLUSION 

 Complete preprocessing: null handling, encoding, normalization  
 Strong correlations found between mileage, horsepower, engine size, and car price  
 Polynomial regression best for price prediction  
 Accident history predicted using classification models with SMOTE  
 Decision Tree gave the best accuracy + visual interpretability  
 Two predictions achieved:
    - Car Price (Regression)
    - Accident History (Classification)

"""
