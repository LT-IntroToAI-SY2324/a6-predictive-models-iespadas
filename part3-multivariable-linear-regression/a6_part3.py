import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Imports and formats the data
# Assuming 'car_data.csv' is in the same directory as your script
data = pd.read_csv("car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(x_train, y_train)

# Find and print the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_
r_squared_train = model.score(x_train, y_train) # R-squared for training data
r_squared_test = r2_score(y_test, model.predict(x_test)) # R-squared for testing data

# Printing the coefficients, intercept, and R squared values rounded to two decimal places
print(f'Coefficients: {np.round(coefficients, 2)}')
print(f'Intercept: {np.round(intercept, 2)}')
print(f'R-squared (training): {np.round(r_squared_train, 2)}')
print(f'R-squared (testing): {np.round(r_squared_test, 2)}')

# Loop through the data and print out the predicted prices and the actual prices
y_pred = model.predict(x_test)
print("***************")
print("Testing Results")
for predicted_price, actual_price in zip(y_pred, y_test):
    print(f'Predicted Price: {np.round(predicted_price, 2)}, Actual Price: {np.round(actual_price, 2)}')
