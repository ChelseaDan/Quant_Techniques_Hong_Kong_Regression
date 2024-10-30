# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Generate some sample data
np.random.seed(42)
data_size = 100
X1 = np.random.rand(data_size) * 10   # Feature 1
X2 = np.random.rand(data_size) * 20   # Feature 2
y = 2.5 * X1 + 1.5 * X2 + np.random.randn(data_size) * 5  # Target variable with some noise

# Create a DataFrame
data = pd.DataFrame({
    'Feature1': X1,
    'Feature2': X2,
    'Target': y
})

# Define features and target
X = data[['Feature1', 'Feature2']]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
