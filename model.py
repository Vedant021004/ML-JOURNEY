import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib

# Load Dataset
df = pd.read_excel("50_Startups_Sample.xlsx")

print(df.head())

# Features
X = df[[
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]]

# Target
y = df["Profit"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train
model.fit(X_train, y_train) # 80% of teh input data

# Predict
prediction = model.predict(X_test) # 20 % data

print("Predictions:")
print(prediction)

# Evaluation
mae = mean_absolute_error(y_test, prediction) #unknown 20% data(y_test) and pridicted data(pridiction)
r2 = r2_score(y_test, prediction) # comparing the scores

print("MAE :", mae)
print("R² Score :", r2)

result = pd.DataFrame({
    "Actual Profit": y_test.values,
    "Predicted Profit": prediction
})

print(result)


# def predict_profit(rd, admin, marketing):

#     features = np.array([[rd, admin, marketing]])

#     predicted_profit = model.predict(features)

#     return predicted_profit[0]


# rd = float(input("Enter R&D Spend: "))
# admin = float(input("Enter Administration Cost: "))
# marketing = float(input("Enter Marketing Spend: "))

# profit = predict_profit(rd, admin, marketing)

# print(f"\nPredicted Profit = ${profit:,.2f}")



joblib.dump(model, "startup_profit_model.pkl")