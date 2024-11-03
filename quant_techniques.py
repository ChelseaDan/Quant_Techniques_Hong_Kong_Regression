# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

#ADd Hong Kong Rental growth data
data_size = 40
delta_vac = [1,2,2,3,4,5]  # Feature 1
delta_gdp = [18,25,50,68,75,65]   # Feature 2
delta_y = [29,25,21,18,15,15]

# Create a DataFrame
data = pd.DataFrame({
    'delta_vac': delta_vac,
    'delta_gdp': delta_gdp,
    'delta_y': delta_y
})

# Define features and target
X = data[['delta_vac', 'delta_gdp']]
y = data['delta_y']


# Initialize and fit the regression model
model = LinearRegression()
model.fit(X, y)

print("Features used:", X.columns)

y_pred = model.predict(X)

# Set up a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(data['delta_vac'], data['delta_gdp'], y, color='blue', label='Data Points')

# Plot the regression plane
# Create a meshgrid for delta_vac and delta_gdp
delta_vac_range = np.linspace(data['delta_vac'].min(), data['delta_vac'].max(), 20)
delta_gdp_range = np.linspace(data['delta_gdp'].min(), data['delta_gdp'].max(), 20)
delta_vac_grid, delta_gdp_grid = np.meshgrid(delta_vac_range, delta_gdp_range)
# Predict values across the meshgrid
y_pred_grid = model.predict(np.c_[delta_vac_grid.ravel(), delta_gdp_grid.ravel()]).reshape(delta_vac_grid.shape)

# Plot the surface for the regression plane
ax.plot_surface(delta_vac_grid, delta_gdp_grid, y_pred_grid, color='red', alpha=0.5, rstride=100, cstride=100)

# Label the axes
ax.set_xlabel('Vacancy Rate Change (delta_vac)')
ax.set_ylabel('GDP % Change (delta_gdp)')
ax.set_zlabel('Office Rent % Change (delta_y)')
ax.set_title('3D Plot of Regression Plane with Data Points')

# Add legend
plt.legend()
plt.show()

# Print the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
