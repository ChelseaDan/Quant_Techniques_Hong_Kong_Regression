# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import statsmodels.stats.diagnostic as smd
from colorama import Fore, Back, Style
from sklearn.metrics import mean_absolute_error, mean_squared_error


#Add Hong Kong Rental growth data

delta_vac_1 = [-45.95, -8.33, -50.91, 96.30, 15.09, 50.82, 5.43, -30.93, 46.27, -4.08, 19.15, 2.68, 38.26, -11.95, -27.14, 8.82, 13.51, 11.11, -9.29, -31.50, -11.49, 15.58, -5.62, 22.62, -22.33, -18.75, -7.69, 16.67, -10.00, 26.98, 2.50, 15.85, -9.47, 4.65, 27.78, 6.96, 17.07]
dummy_1 = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
delta_unemployment_1 = [-39.29, -17.65, -21.43, 18.18, 38.46, 8.89, 0.00, -3.06, 69.47, -12.11, -21.55, 106.31, 36.46, -21.28, 3.46, 43.03, 7.97, -14.25, -17.21, -14.52, -15.93, -11.22, 47.75, -17.70, -20.95, -4.00, 3.50, -3.06, 0.55, 2.35, -7.90, -10.21, 3.99, 99.11, -10.92, -16.441, -9.028]
delta_gdp_1 = [23.24, 17.95, 15.21, 11.83, 15.64, 17.21, 15.42, 12.84, 6.51, 10.41, 11.04, -4.77, -1.85, 3.56, -1.32, -1.80, -2.98, 4.78, 7.37, 6.59, 9.33, 3.63, -2.39, 6.82, 8.69, 5.68, 4.98, 5.72, 6.15, 3.70, 6.36, 5.99, 0.37, -4.98, 6.95, -2.7690, 6.5120]
delta_y_1 = [27.73, 27.06, 64.14, 0.66, -5.61, 5.56, 9.58, 21.28, -1.76, -14.73, 2.95, -13.33, -26.42, -1.50, 2.54, -15.45, -12.65, 4.69, 23.43, 21.78, 12.35, 17.89, -12.73, 8.77, 15.11, 10.83, 8.39, 4.70, 6.08, 2.47, 4.09, 4.30, 3.65, -7.54, -3.43, -1.457, -1.000]

delta_gdp_2 = [17.95, 15.21, 11.83, 15.64, 17.21, 15.42, 12.84, 6.51, 10.41, 11.04, -4.77, -1.85, 3.56, -1.32, -1.80, -2.98, 4.78, 7.37, 6.59, 9.33, 3.63, -2.39, 6.82, 8.69, 5.68, 4.98, 5.72, 6.15, 3.70, 6.36, 5.99, 0.37, -4.98, 6.95, -2.7690, 6.5120]
delta_unemployment_2 = [-17.65, -21.43, 18.18, 38.46, 8.89, 0.00, -3.06, 69.47, -12.11, -21.55, 106.31, 36.46, -21.28, 3.46, 43.03, 7.97, -14.25, -17.21, -14.52, -15.93, -11.22, 47.75, -17.70, -20.95, -4.00, 3.50, -3.06, 0.55, 2.35, -7.90, -10.21, 3.99, 99.11, -10.92, -16.441, -9.028]
delta_stock_2 = [0.88, 0.14, 5.36, 5.14, 3.71, 8.73, 10.45, 5.53, 6.6, 4.77, 2.77, 6.69, 9.54, 4.1, 0.97, 0.95, 1.36, 2.72, 2.68, -0.26, 0.44, 3.0, 2.83, 1.32, 1.52, 0.87, 1.01, 0.85, 0.71, 2.01, 2.19, 2.67, 1.82, 2.16, 0.92, 0.91]
delta_y_2 = [27.06, 64.14, 0.66, -5.61, 5.56, 9.58, 21.28, -1.76, -14.73, 2.95, -13.33, -26.42, -1.50, 2.54, -15.45, -12.65, 4.69, 23.43, 21.78, 12.35, 17.89, -12.73, 8.77, 15.11, 10.83, 8.39, 4.70, 6.08, 2.47, 4.09, 4.30, 3.65, -7.54, -3.43, -1.457, -1.000]
dummy_2 = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

delta_vac_forecast = [-9.47, 4.65, 27.78, 6.96, 17.07]
delta_gdp_forecast = [0.37, -4.98, 6.95, -2.77, 6.51]
delta_stock_forecast = [2.16, 0.92, 0.91, 2.97, 1.53]
delta_unemployment_forecast = [3.99, 99.11, -10.92, -16.54, -8.96]
delta_y_forecast = [3.65, -7.54, -3.43, -1.46, 1.00]

print("Checking data lengths")
print(len(set([len(delta_vac_1), len(dummy_1), len(delta_unemployment_1), len(delta_gdp_1), len(delta_y_1)])) == 1)
print(len(set([len(delta_stock_2), len(dummy_2), len(delta_unemployment_2), len(delta_gdp_2), len(delta_y_2)])) == 1)


#data = pd.DataFrame({
#    'delta_y': delta_y_1,
#    'delta_gdp': delta_gdp_1,
#    'delta_unemployment': delta_unemployment_1,
#    'delta_vac': delta_vac_1
#    'delta_stock': delta_stock_test
#})

# Adjust pandas settings to display the entire table
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# Add lagged and lead variables for up to 3 periods
#lags_leads = [3, 2, 1, 0]
#correlation_table = pd.DataFrame(index=lags_leads)

# Calculate correlations
#for col in ['delta_stock', 'delta_vac', 'delta_gdp', 'delta_unemployment']:
#    for lag_lead in lags_leads:
#        shifted_col = data[col].shift(lag_lead)
#        correlation = data['delta_y'].corr(shifted_col)
#        correlation_table.loc[lag_lead, f"{col}_lag{lag_lead}" if lag_lead < 0 else f"{col}_t{lag_lead}"] = correlation

# Display the correlation table
#correlation_table = correlation_table.sort_index()
#print("\nCorrelation Table (lags up to 3 periods):")
#print(correlation_table)

# Plot heatmap of the correlation table for better visualization
#plt.figure(figsize=(12, 8))
#plt.imshow(correlation_table, cmap='coolwarm', aspect='auto', interpolation='none')
#plt.colorbar(label='Correlation Coefficient')
#plt.title('Heatmap of Correlations Between Variables and Delta_Y', fontsize=16)
#plt.xlabel('Variables (lags)', fontsize=14)
#plt.ylabel('Lag Periods', fontsize=14)
#plt.xticks(range(correlation_table.shape[1]), correlation_table.columns, rotation=90, fontsize=12)
#plt.yticks(range(correlation_table.shape[0]), correlation_table.index, fontsize=12)
#plt.grid(False)
#plt.tight_layout()
#plt.show()

# Create a DataFrame
data_1 = pd.DataFrame({
    'delta_unemployment': delta_unemployment_1,
    'delta_gdp': delta_gdp_1,
    'dummy': dummy_1,
    'delta_vac': delta_vac_1,
    'delta_y': delta_y_1
})

data_2 = pd.DataFrame({
    'delta_stock': delta_stock_2,
    'delta_unemployment': delta_unemployment_2,
    'delta_gdp': delta_gdp_2,
    'dummy': dummy_2,
    'delta_y': delta_y_2
})

# Add interaction terms to your data
#data['interaction1'] = data['delta_vac'] * data['delta_gdp']
#data['interaction2'] = data['delta_gdp'] * data['delta_unemployment']
#data['interaction3'] = data['delta_vac'] * data['delta_unemployment']

# Adjust display options
#pd.set_option('display.max_rows', None)  # Show all rows
#pd.set_option('display.max_columns', None)  # Show all columns

# Correlation matrix
#correlation_matrix = data_1.corr()
#print(correlation_matrix)
#print('\n\n')

# Correlation matrix
#correlation_matrix = data_2.corr()
#print(correlation_matrix)
#print('\n\n')

# Define features and target

X_1 = data_1[['dummy', 'delta_vac', 'delta_gdp', 'delta_unemployment']]
X_2 = data_2[['delta_gdp', 'dummy', 'delta_unemployment', 'delta_stock']]

X_1 = sm.add_constant(X_1)
X_2 = sm.add_constant(X_2)
y_1 = data_1['delta_y']
y_2 = data_2['delta_y']

# Fit the model with statsmodels to get p-values
model_1 = sm.OLS(y_1, X_1)
model_2 = sm.OLS(y_2, X_2)

results_1 = model_1.fit()
results_2 = model_2.fit()

# Perform the RESET test
reset_test = smd.linear_reset(results_1, power=2, use_f=True)

# Display the results
print("RESET Test Results Model 1:")
print(f"F-statistic: {reset_test.fvalue}")
print(Fore.GREEN + f"P-value: {reset_test.pvalue}")
print(Style.RESET_ALL)

# Perform the RESET test
#reset_test = smd.linear_reset(results_2, power=2, use_f=True)

# Display the results
print("\nRESET Test Results Model 2:")
print(f"F-statistic: {reset_test.fvalue}")
print(Fore.GREEN + f"P-value: {reset_test.pvalue}")
print(Style.RESET_ALL)

# Print summary, which includes coefficients, standard errors, t-statistics, and p-values
print(results_1.summary())
print(results_2.summary())

# Perform White's Test for heteroscedasticity
white_test = het_white(results_1.resid, X_1)

# Extract test results
lm_stat, lm_pval, f_stat, f_pval = white_test

print("\nWhite's Test for Heteroscedasticity:")
print(f"Lagrange Multiplier Statistic: {lm_stat}")
print(Fore.GREEN + f"p-value (LM test): {lm_pval}")
print(Style.RESET_ALL)
print(f"F-statistic: {f_stat}")
print(Fore.GREEN + f"p-value (F-test): {f_pval}")
print(Style.RESET_ALL)

# Perform White's Test for heteroscedasticity
white_test = het_white(results_2.resid, X_2)

# Extract test results
lm_stat, lm_pval, f_stat, f_pval = white_test

print("\nWhite's Test for Heteroscedasticity:")
print(f"Lagrange Multiplier Statistic: {lm_stat}")
print(Fore.GREEN + f"p-value (LM test): {lm_pval}")
print(Style.RESET_ALL)
print(f"F-statistic: {f_stat}")
print(Fore.GREEN + f"p-value (F-test): {f_pval}")
print(Style.RESET_ALL)

# Interpretation
if lm_pval < 0.05:
    print("Reject the null hypothesis: Evidence of heteroscedasticity.")
else:
    print(Fore.GREEN + "Fail to reject the null hypothesis: No evidence of heteroscedasticity.")
    print(Style.RESET_ALL)

vac_series = pd.Series(delta_vac_1)
unemployment_series = pd.Series(delta_unemployment_1)
gdp_series = pd.Series(delta_gdp_1)
y_series = pd.Series(delta_y_1)
stock_series = pd.Series(delta_stock_2)

vif_data_1 = pd.DataFrame()
vif_data_1['Feature'] = X_1.columns
vif_data_1['VIF'] = [variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]

vif_data_2 = pd.DataFrame()
vif_data_2['Feature'] = X_2.columns
vif_data_2['VIF'] = [variance_inflation_factor(X_2.values, i) for i in range(X_2.shape[1])]

print("\nVariance Inflation Factor (VIF) for Multicollinearity for Model 1:")
print(vif_data_1)

print("\nVariance Inflation Factor (VIF) for Multicollinearity for Model 2:")
print(vif_data_2)

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

# Run ADF test on stock_series
result_stock = adfuller(stock_series)
print("\nStock Series ADF Statistic:", result_stock[0])
print("p-value:", result_stock[1])
print("Critical Values:", result_stock[4])

# Extract coefficients and intercept from the regression model
intercept_1 = results_1.params['const']
coefficients_1 = results_1.params[['delta_gdp', 'delta_vac', 'delta_unemployment']]

intercept_2 = results_2.params['const']
coefficients_2 = results_2.params[['delta_stock', 'delta_gdp', 'delta_unemployment']]

from scipy.stats import f

def chow_test(data, split_index, dependent_var, independent_vars):
    """
    Perform the Chow test for structural breaks.

    Args:
        data (pd.DataFrame): Dataset containing the variables.
        split_index (int): Index to split the data into two parts.
        dependent_var (str): The dependent variable column name.
        independent_vars (list): List of independent variable column names.

    Returns:
        F-statistic, p-value
    """
    # Split the dataset
    data1 = data.iloc[:split_index]
    data2 = data.iloc[split_index:]
    
    # Fit the models on both subsets
    X1 = sm.add_constant(data1[independent_vars])
    y1 = data1[dependent_var]
    model1 = sm.OLS(y1, X1).fit()
    
    X2 = sm.add_constant(data2[independent_vars])
    y2 = data2[dependent_var]
    model2 = sm.OLS(y2, X2).fit()
    
    # Combined model
    X_combined = sm.add_constant(data[independent_vars])
    y_combined = data[dependent_var]
    combined_model = sm.OLS(y_combined, X_combined).fit()
    
    # Calculate the Chow test statistic
    rss1 = model1.ssr
    rss2 = model2.ssr
    rss_combined = combined_model.ssr
    
    k = len(independent_vars) + 1  # Number of predictors including the intercept
    n1 = len(data1)
    n2 = len(data2)
    
    F_stat = ((rss_combined - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
    p_value = f.sf(F_stat, k, n1 + n2 - 2 * k)
    
    return F_stat, p_value


split_index_1 = len(data_1) // 2  # Split the dataset into two equal parts
split_index_2 = len(data_2) // 2
dependent_var = 'delta_y'
independent_vars_1 = ['delta_gdp', 'delta_vac', 'delta_unemployment']
independent_vars_2 = ['delta_gdp', 'delta_unemployment', 'delta_stock']

F_stat, p_value = chow_test(data_1, split_index_1, dependent_var, independent_vars_1)
print("\nChow Test Results 1:")
print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {p_value:.4f}")

F_stat, p_value = chow_test(data_2, split_index_2, dependent_var, independent_vars_2)
print("\nChow Test Results 2:")
print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {p_value:.4f}")

forecast_data_1 = pd.DataFrame({
    'delta_gdp': delta_gdp_forecast,
    'delta_unemployment': delta_unemployment_forecast,
    'delta_vac': delta_vac_forecast,
    'delta_y_actual': delta_y_forecast
})

forecast_data_2 = pd.DataFrame({
    'delta_stock': delta_stock_forecast,
    'delta_unemployment': delta_unemployment_forecast,
    'delta_gdp': delta_gdp_forecast,
    'delta_y_actual': delta_y_forecast
})

forecast_data_1['delta_y_predicted_1'] = (
    intercept_1 +
    coefficients_1['delta_vac'] * forecast_data_1['delta_vac'] +
    coefficients_1['delta_gdp'] * forecast_data_1['delta_gdp'] +
    coefficients_1['delta_unemployment'] * forecast_data_1['delta_unemployment']
)

forecast_data_2['delta_y_predicted_2'] = (
    intercept_2 +
    coefficients_2['delta_stock'] * forecast_data_2['delta_stock'] +
    coefficients_2['delta_gdp'] * forecast_data_2['delta_gdp'] +
    coefficients_2['delta_unemployment'] * forecast_data_2['delta_unemployment']
)

# Display the forecast table
print(forecast_data_1)
print(forecast_data_2)

years = ['2019', '2020', '2021', '2022', '2023']
# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(years, delta_y_forecast, marker='o', label='Actual Y', linestyle='-', linewidth=2)
plt.plot(years, forecast_data_1['delta_y_predicted_1'], marker='s', label='Predicted Y (1)', linestyle='--', linewidth=2)
plt.plot(years, forecast_data_2['delta_y_predicted_2'], marker='x', label='Predicted Y (2)', linestyle='solid', linewidth=2)

# Adding labels, title, and legend
plt.title('Comparison of Actual and Predicted Y Values (Forecast)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Hong Kong Rental Growth Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Calculate forecast errors for Model 1
actual_1 = forecast_data_1['delta_y_actual']
predicted_1 = forecast_data_1['delta_y_predicted_1']

# Calculate forecast errors for Model 2
actual_2 = forecast_data_2['delta_y_actual']
predicted_2 = forecast_data_2['delta_y_predicted_2']

# Metrics for Model 1
mae_1 = mean_absolute_error(actual_1, predicted_1)
mse_1 = mean_squared_error(actual_1, predicted_1)
rmse_1 = np.sqrt(mse_1)
mape_1 = np.mean(np.abs((actual_1 - predicted_1) / actual_1)) * 100

# Metrics for Model 2
mae_2 = mean_absolute_error(actual_2, predicted_2)
mse_2 = mean_squared_error(actual_2, predicted_2)
rmse_2 = np.sqrt(mse_2)
mape_2 = np.mean(np.abs((actual_2 - predicted_2) / actual_2)) * 100

# Print the metrics for both models
print("\nModel 1 Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae_1:.4f}")
print(f"Mean Squared Error (MSE): {mse_1:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_1:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_1:.2f}%")

print("\nModel 2 Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"Mean Squared Error (MSE): {mse_2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_2:.2f}%")