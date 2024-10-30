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
delta_vac = [-45.95, -8.33, -50.91, 96.30, 15.09, 50.82, 5.43, -30.93, 46.27, -4.08, 19.15, 2.68, 38.26, -11.95, -27.14, 8.82, 13.51, 11.11, -9.29, -31.50, -11.49, 15.58, -5.62, 22.62, -22.33, -18.75, -7.69, 16.67, -10.00, 26.98, 2.50, 15.85, -9.47, 4.65, 27.78, 6.96, 17.07, 3.47]  # Feature 1
delta_gdp = [0.150578684, 0.232438394, 0.179454537, 0.152122081, 0.118309922, 0.156394168, 0.172128036, 0.15422766, 0.128433881, 0.065093948, 0.104152431, 0.110413172, -0.047745022, -0.01845836, 0.035596692, -0.013191462, -0.018036502, -0.029836737, 0.047799878, 0.073740745, 0.06590393, 0.093324, 0.036303949, -0.023855269, 0.068166425, 0.08692733, 0.056798691, 0.049758483, 0.057175533, 0.061502871, 0.03708865, 0.063619497, 0.059945452, 0.003714013, -0.049938493, 0.06960863, -0.027802663, 0.065120034]   # Feature 2
delta_y = [16.60, 35.92, 21.19, 32.52, -3.97, 1.21, 36.69, 20.06, 39.91, -15.50, -3.19, 13.11, -36.88, -25.65, -10.10, -12.46, -13.09, -8.63, 28.88, 33.94, 4.74, 18.81, 20.24, -9.65, 28.14, 29.30, 12.35, 22.44, 3.22, 6.12, -4.90, 14.10, 13.88, -2.11, -13.66, 7.19, -1.35, -5.45]

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
