# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


#Add Hong Kong Rental growth data

vac = [11.1, 6.0, 5.5, 2.7, 5.3, 6.1, 9.2, 9.7, 6.7, 9.8, 9.4, 11.2, 11.5, 15.9, 14.0, 10.2, 11.1, 12.6, 14.0, 12.7, 8.7, 7.7, 8.9, 8.4, 10.3, 8.0, 6.5, 6.0, 7.0, 6.3, 8.0, 8.2, 9.5, 8.6, 9.0, 11.5, 12.3, 14.4]

delta_vac = [-20,-45.95, -8.33, -50.91, 96.30, 15.09, 50.82, 5.43, -30.93, 46.27, -4.08, 19.15, 2.68, 38.26, -11.95, -27.14, 8.82, 13.51, 11.11, -9.29, -31.50, -11.49, 15.58, -5.62, 22.62, -22.33, -18.75, -7.69, 16.67, -10.00, 26.98, 2.50, 15.85, -9.47, 4.65, 27.78, 6.96, 17.07]
cpi = [2.75, 3.3846, 5.6548, 7.8873, 10.1828, 10.4265, 11.1588, 9.6525, 8.8028, 8.7379, 9.0774, 6.2756, 5.7766, 2.9126, -4.0094, -3.6855, -1.6582, -2.9831, -2.6738, -0.2747, 0.8264, 2.0003, 2.035, 4.3032, 0.5794, 2.2929, 5.3052, 4.0539, 4.338, 4.4236, 2.9908, 2.4093, 1.4939, 2.4061, 2.8832, 0.251, 1.5688, 1.8814]
gdp = [35699543050.78, 41075570591.93, 50622571586.11, 59707404560.59, 68790369107.30, 76928290841.87, 88959620135.89, 104272278634.73, 120353947980.76, 135812069768.65, 144652912433.10, 159717233621.66, 177352785419.98, 168886163221.57, 165768095391.56, 171668164082.56, 169403241524.34, 166349228737.39, 161384522525.30, 169099768875.19, 181570082162.19, 193536265094.36, 211597405593.87, 219279678430.16, 214046415026.19, 228637697575.04, 248513617677.29, 262629441493.48, 275696879834.97, 291459356985.34, 309383627028.56, 320837638328.85, 341244161576.76, 361691522612.75, 363016373358.52, 344932192028.05, 368911387845.42, 358696261481.16, 382054574298.53
]
dummy = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#https://ycharts.com/indicators/hong_kong_unemployment_rate_annual
#https://www.macrotrends.net/global-metrics/countries/HKG/hong-kong/unemployment-rate#:~:text=Hong%20Kong%20unemployment%20rate%20for,a%202.89%25%20increase%20from%202019.
unemployment = [2.8, 1.7, 1.4, 1.1, 1.3, 1.8, 1.96, 1.96, 1.9, 3.22, 2.83, 2.22, 4.58, 6.25, 4.92, 5.09, 7.28, 7.86, 6.74, 5.58, 4.77, 4.01, 3.56, 5.26, 4.329, 3.422, 3.285, 3.4, 3.296, 3.314, 3.392, 3.124, 2.805, 2.917, 5.808, 5.174, 4.318, 3.931]
delta_unemployment = [-12.50, -39.29, -17.65, -21.43, 18.18, 38.46, 8.89, 0.00, -3.06, 69.47, -12.11, -21.55, 106.31, 36.46, -21.28, 3.46, 43.03, 7.97, -14.25, -17.21, -14.52, -15.93, -11.22, 47.75, -17.70, -20.95, -4.00, 3.50, -3.06, 0.55, 2.35, -7.90, -10.21, 3.99, 99.11, -10.92, -16.54, -8.96]
delta_gdp = [0.150591, 23.24, 17.95, 15.21, 11.83, 15.64, 17.21, 15.42, 12.84, 6.51, 10.41, 11.04, -4.77, -1.85, 3.56, -1.32, -1.80, -2.98, 4.78, 7.37, 6.59, 9.33, 3.63, -2.39, 6.82, 8.69, 5.68, 4.98, 5.72, 6.15, 3.70, 6.36, 5.99, 0.37, -4.98, 6.95, -2.77, 6.51]
delta_y = [17.97, 27.73, 27.06, 64.14, 0.66, -5.61, 5.56, 9.58, 21.28, -1.76, -14.73, 2.95, -13.33, -26.42, -1.50, 2.54, -15.45, -12.65, 4.69, 23.43, 21.78, 12.35, 17.89, -12.73, 8.77, 15.11, 10.83, 8.39, 4.70, 6.08, 2.47, 4.09, 4.30, 3.65, -7.54, -3.43, -1.46, 1.00]

print(len(delta_vac))
print(len(cpi))
print(len(delta_unemployment))
print(len(delta_gdp))
print(len(dummy))
print(len(delta_y))

# Create a DataFrame
data = pd.DataFrame({
    'delta_vac': delta_vac,
    'delta_unemployment': delta_unemployment,
    'delta_gdp': delta_gdp,
    'dummy': dummy,
    'delta_y': delta_y
})

# Correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)


# Define features and target
X = data[['delta_vac', 'delta_gdp','dummy','delta_unemployment']]
X = sm.add_constant(X)
y = data['delta_y']

# Fit the model with statsmodels to get p-values
model = sm.OLS(y, X)
results = model.fit()

# Print summary, which includes coefficients, standard errors, t-statistics, and p-values
print(results.summary())

vac_series = pd.Series(delta_vac)
unemployment_series = pd.Series(delta_unemployment)
gdp_series = pd.Series(delta_gdp)

result_vac = adfuller(vac_series)

#Run ADF test on vacancy_series
print("Vacancy Series ADF Statistic:", result_vac[0])
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
