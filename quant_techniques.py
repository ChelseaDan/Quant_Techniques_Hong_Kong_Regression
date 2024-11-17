# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


#Add Hong Kong Rental growth data

delta_vac_test = [-20,-45.95, -8.33, -50.91, 96.30, 15.09, 50.82, 5.43, -30.93, 46.27, -4.08, 19.15, 2.68, 38.26, -11.95, -27.14, 8.82, 13.51, 11.11, -9.29, -31.50, -11.49, 15.58, -5.62, 22.62, -22.33, -18.75, -7.69, 16.67, -10.00, 26.98, 2.50, 15.85]
dummy_test = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
delta_unemployment_test = [-12.50, -39.29, -17.65, -21.43, 18.18, 38.46, 8.89, 0.00, -3.06, 69.47, -12.11, -21.55, 106.31, 36.46, -21.28, 3.46, 43.03, 7.97, -14.25, -17.21, -14.52, -15.93, -11.22, 47.75, -17.70, -20.95, -4.00, 3.50, -3.06, 0.55, 2.35, -7.90, -10.21]
delta_gdp_test = [15.0591, 23.24, 17.95, 15.21, 11.83, 15.64, 17.21, 15.42, 12.84, 6.51, 10.41, 11.04, -4.77, -1.85, 3.56, -1.32, -1.80, -2.98, 4.78, 7.37, 6.59, 9.33, 3.63, -2.39, 6.82, 8.69, 5.68, 4.98, 5.72, 6.15, 3.70, 6.36, 5.99]
delta_y_test = [17.97, 27.73, 27.06, 64.14, 0.66, -5.61, 5.56, 9.58, 21.28, -1.76, -14.73, 2.95, -13.33, -26.42, -1.50, 2.54, -15.45, -12.65, 4.69, 23.43, 21.78, 12.35, 17.89, -12.73, 8.77, 15.11, 10.83, 8.39, 4.70, 6.08, 2.47, 4.09, 4.30]

delta_vac_forecast = [-9.47, 4.65, 27.78, 6.96, 17.07]
delta_gdp_forecast = [0.37, -4.98, 6.95, -2.77, 6.51]
delta_unemployment_forecast = [3.99, 99.11, -10.92, -16.54, -8.96]
delta_y_forecast = [.65, -7.54, -3.43, -1.46, 1.00]

#print(len(delta_vac))
#print(len(cpi))
print(len(delta_gdp_test))
print(len(delta_vac_test))
print(len(delta_y_test))
print(len(delta_unemployment_test))
#print(len(delta_gdp))
print(len(dummy_test))
#print(len(delta_y))

# Create a DataFrame
data = pd.DataFrame({
    'delta_vac': delta_vac_test,
    'delta_unemployment': delta_unemployment_test,
    'delta_gdp': delta_gdp_test,
    'dummy': dummy_test,
    'delta_y': delta_y_test
})

# Add interaction terms to your data
#data['interaction1'] = data['delta_vac'] * data['delta_gdp']
#data['interaction2'] = data['delta_gdp'] * data['delta_unemployment']
#data['interaction3'] = data['delta_vac'] * data['delta_unemployment']

# Correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)
print('\n\n')

# Define features and target

X = data[['delta_vac', 'delta_gdp','dummy','delta_unemployment']]
X = sm.add_constant(X)
y = data['delta_y']

# Fit the model with statsmodels to get p-values
model = sm.OLS(y, X)
results = model.fit()

# Print summary, which includes coefficients, standard errors, t-statistics, and p-values
print(results.summary())

# Perform White's Test for heteroscedasticity
white_test = het_white(results.resid, X)

# Extract test results
lm_stat, lm_pval, f_stat, f_pval = white_test

print("\nWhite's Test for Heteroscedasticity:")
print(f"Lagrange Multiplier Statistic: {lm_stat}")
print(f"p-value (LM test): {lm_pval}")
print(f"F-statistic: {f_stat}")
print(f"p-value (F-test): {f_pval}")

# Interpretation
if lm_pval < 0.05:
    print("Reject the null hypothesis: Evidence of heteroscedasticity.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

vac_series = pd.Series(delta_vac_test)
unemployment_series = pd.Series(delta_unemployment_test)
gdp_series = pd.Series(delta_gdp_test)
y_series = pd.Series(delta_y_test)

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF) for Multicollinearity:")
print(vif_data)

# Interpretation of VIF
print("\nVIF Interpretation:")
print(" - VIF > 10: High multicollinearity (problematic)")
print(" - VIF between 5 and 10: Moderate multicollinearity (may need attention)")
print(" - VIF < 5: Low multicollinearity (acceptable)\n\n")

result_y = adfuller(y_series)
#Run ADF test on y
print("Y Series ADF Statistic:", result_y[0])
print("p-value:", result_y[1])
print("Critical Values:", result_y[4])

result_vac = adfuller(vac_series)
#Run ADF test on vacancy_series
print("\nVacancy Series ADF Statistic:", result_vac[0])
print("p-value:", result_vac[1])
print("Critical Values:", result_vac[4])

# Run ADF test on unemployment_series
result_unemployment = adfuller(unemployment_series)
print("\nUnemployment Series ADF Statistic:", result_unemployment[0])
print("p-value:", result_unemployment[1])
print("Critical Values:", result_unemployment[4])

# Run ADF test on gdp_series
result_gdp = adfuller(gdp_series)
print("\nGDP Series ADF Statistic:", result_gdp[0])
print("p-value:", result_gdp[1])
print("Critical Values:", result_gdp[4])

# Extract coefficients and intercept from the regression model
intercept = results.params['const']
coefficients = results.params[['delta_vac', 'delta_gdp', 'delta_unemployment']]

forecast_data = pd.DataFrame({
    'delta_vac': delta_vac_forecast,
    'delta_gdp': delta_gdp_forecast,
    'delta_unemployment': delta_unemployment_forecast,
    'delta_y_actual': delta_y_forecast
})

forecast_data['delta_y_predicted'] = (
    intercept +
    coefficients['delta_vac'] * forecast_data['delta_vac'] +
    coefficients['delta_gdp'] * forecast_data['delta_gdp'] +
    coefficients['delta_unemployment'] * forecast_data['delta_unemployment']
)

# Display the forecast table
print(forecast_data)

years = ['2019', '2020', '2021', '2022', '2023']
# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(years, delta_y_forecast, marker='o', label='Actual Y', linestyle='-', linewidth=2)
plt.plot(years, forecast_data['delta_y_predicted'], marker='s', label='Predicted Y', linestyle='--', linewidth=2)

# Adding labels, title, and legend
plt.title('Comparison of Actual and Predicted Y Values (Forecast)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Hong Kong Rental Growth Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()