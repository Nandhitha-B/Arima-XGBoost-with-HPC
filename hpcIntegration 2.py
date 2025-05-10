
import dask.dataframe as dd
import xgboost as xgb
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dask.distributed import Client
import matplotlib.pyplot as plt
import time

start_time_program = time.time()

client = Client()


data = yf.download("AAPL", start="2015-01-01", end="2025-01-01")
data = data[['Close']]
data.index = pd.to_datetime(data.index)
data = data.asfreq('B').ffill()

indicators = pd.DataFrame()
indicators['Return'] = data['Close'].pct_change()
indicators['MA_5'] = data['Close'].rolling(window=5).mean()
indicators['MA_20'] = data['Close'].rolling(window=20).mean()
indicators['Lag_1'] = data['Close'].shift(1)
indicators = indicators.dropna()

indicators = dd.from_pandas(indicators, npartitions=4)

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]



def fit_arima_model(train_data):
    model = ARIMA(train_data, order=(1, 0, 1))
    model_fit = model.fit()
    return model_fit

start_time_arima = time.time()
futures = client.map(fit_arima_model, [train])
model_fit = client.gather(futures)[0]

indicators['ARIMA_Pred'] = model_fit.predict(start=data.index[0], end=data.index[-1])
end_time_arima = time.time()
indicators_test_aligned = indicators.loc[test.index]
arima_execution_time = end_time_arima - start_time_arima

arima_rmse = np.sqrt(mean_squared_error(test['Close'], indicators_test_aligned['ARIMA_Pred']))
print(f'RMSE for ARIMA: {arima_rmse}')

start_time_xgboost = time.time()
X = indicators[['Return', 'MA_5', 'MA_20', 'Lag_1', 'ARIMA_Pred']].compute()
y = data['Close'].loc[indicators.index]

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_dask = dd.from_pandas(X_train, npartitions=4)
y_train_dask = dd.from_pandas(y_train, npartitions=4)
X_test_dask = dd.from_pandas(X_test, npartitions=4)

dask_model = xgb.dask.DaskXGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
dask_model.fit(X_train_dask, y_train_dask)

y_pred_xgb = dask_model.predict(X_test_dask)
y_pred_xgb = y_pred_xgb.compute()

indicators = indicators.compute()
indicators['XGBoost_Pred'] = pd.Series(np.nan, index=indicators.index)
indicators.loc[indicators.index[train_size:], 'XGBoost_Pred'] = y_pred_xgb
indicators_test_aligned = indicators.loc[test.index]
end_time_xgboost = time.time()
xgboost_execution_time = end_time_xgboost - start_time_xgboost
xgboost_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'RMSE for XGBoost: {xgboost_rmse}')

start_time_hybrid = time.time()
X_stack = indicators_test_aligned[['ARIMA_Pred', 'XGBoost_Pred']].dropna()
y_stack = test['Close'].loc[X_stack.index]
if not X_stack.empty:
    stacking_model = LinearRegression()
    stacking_model.fit(X_stack, y_stack)

    indicators_test_aligned.loc[X_stack.index, 'Hybrid_Pred'] = stacking_model.predict(X_stack)
else:
    print("No valid data for stacking model.")

end_time_hybrid = time.time()
hybrid_execution_time = end_time_hybrid - start_time_hybrid

hybrid_rmse = np.sqrt(mean_squared_error(y_stack, indicators_test_aligned['Hybrid_Pred']))
print(f'RMSE for Hybrid Model: {hybrid_rmse}')
print(indicators_test_aligned[['ARIMA_Pred',
      'XGBoost_Pred', 'Hybrid_Pred']])

indicators_test_aligned['Close'] = test['Close']

end_time_program = time.time()
total_execution_time = end_time_program - start_time_program

plt.figure(figsize=(14, 8))


plt.plot(indicators_test_aligned.index, indicators_test_aligned['Close'], label='Actual Close', color='black', linewidth=2)


plt.plot(indicators_test_aligned.index, indicators_test_aligned['ARIMA_Pred'], label='ARIMA Predictions', linestyle='--', color='blue')


plt.plot(indicators_test_aligned.index, indicators_test_aligned['XGBoost_Pred'], label='XGBoost Predictions', linestyle='--', color='red')


plt.plot(indicators_test_aligned.index, indicators_test_aligned['Hybrid_Pred'], label='Hybrid Predictions', linestyle='-', color='green')

plt.title('Comparison of Model Predictions vs Actual Close Prices', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"ARIMA Execution Time: {arima_execution_time:.4f} seconds")
print(f"XGBoost Execution Time: {xgboost_execution_time:.4f} seconds")
print(f"Hybrid Model Execution Time: {hybrid_execution_time:.4f} seconds")
print(f"Total Program Execution Time: {total_execution_time:.4f} seconds")

performance_results = [
    {
        "Model": "ARIMA",
        "RMSE": arima_rmse,
        "Execution Time (s)": arima_execution_time
    },
    {
        "Model": "XGBoost",
        "RMSE": xgboost_rmse,
        "Execution Time (s)": xgboost_execution_time
    },
    {
        "Model": "Hybrid",
        "RMSE": hybrid_rmse,
        "Execution Time (s)": hybrid_execution_time
    },
    {
        "Model": "Total Program",
        "RMSE": None,
        "Execution Time (s)": total_execution_time
    }
]

summary_df = pd.DataFrame(performance_results)

print("\nPerformance Summary:")
print(summary_df)

client.close()

import yfinance as yf
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

start_time_program_2 = time.time()

data = yf.download("AAPL", start="2015-01-01", end="2025-01-01")
data = data[['Close']]

indicators = pd.DataFrame()

indicators['Return'] = data['Close'].pct_change()
indicators['MA_5'] = data['Close'].rolling(window=5).mean()
indicators['MA_20'] = data['Close'].rolling(window=20).mean()
indicators['Lag_1'] = data['Close'].shift(1)

data.index = pd.to_datetime(data.index)
data = data.asfreq('B').ffill()

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
start_time_arima_norm = time.time()
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()

indicators['ARIMA_Pred'] = model_fit.predict(
    start=data.index[0], end=data.index[-1])
indicators['Close'] = data['Close']
indicators.index = pd.to_datetime(indicators.index)
indicators = indicators.asfreq('B').ffill()
end_time_arima_norm = time.time()


indicators = indicators.dropna()

y = data['Close'].loc[indicators.index]

X = indicators[['Return', 'MA_5', 'MA_20', 'Lag_1', 'ARIMA_Pred']]

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
start_time_xgboost_norm = time.time()

model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)


y_pred_xgb = model_xgb.predict(X_test)
end_time_xgboost_norm = time.time()


indicators_test_aligned = indicators.loc[test.index]

X_test_aligned = X.loc[test.index]

indicators_test_aligned['XGBoost_Pred'] = model_xgb.predict(
    X_test_aligned)

indicators_test_aligned['ARIMA_Pred'] = model_fit.predict(
    start=test.index[0], end=test.index[-1])
start_time_hybrid_norm = time.time()
X_stack = indicators_test_aligned[['ARIMA_Pred', 'XGBoost_Pred']]
y_stack = test['Close']

stacking_model = LinearRegression()
stacking_model.fit(X_stack, y_stack)

indicators_test_aligned['Hybrid_Pred'] = stacking_model.predict(X_stack)
end_time_hybrid_norm = time.time()
print(indicators_test_aligned[['ARIMA_Pred',
      'XGBoost_Pred', 'Hybrid_Pred', 'Close']])



def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

arima_rmse_2 = calculate_rmse(
    test['Close'], indicators_test_aligned['ARIMA_Pred'])
xgboost_rmse_2 = calculate_rmse(
    test['Close'], indicators_test_aligned['XGBoost_Pred'])
hybrid_rmse_2 = calculate_rmse(
    test['Close'], indicators_test_aligned['Hybrid_Pred'])

end_time_program_2 = time.time()

print(f"RMSE for ARIMA: {arima_rmse_2:.4f}")
print(f"RMSE for XGBoost: {xgboost_rmse_2:.4f}")
print(f"RMSE for Hybrid: {hybrid_rmse_2:.4f}")


plt.figure(figsize=(14, 8))


plt.plot(indicators_test_aligned.index,
         indicators_test_aligned['Close'], label='Actual Close', color='black', linewidth=2)


plt.plot(indicators_test_aligned.index,
         indicators_test_aligned['ARIMA_Pred'], label='ARIMA Predictions', linestyle='--', color='blue')


plt.plot(indicators_test_aligned.index,
         indicators_test_aligned['XGBoost_Pred'], label='XGBoost Predictions', linestyle='--', color='red')


plt.plot(indicators_test_aligned.index,
         indicators_test_aligned['Hybrid_Pred'], label='Hybrid Predictions', linestyle='-', color='green')


plt.title('Comparison of Model Predictions vs Actual Close Prices', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)


plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

arima_execution_time_2 = end_time_arima_norm - start_time_arima_norm
xgboost_execution_time_2 = end_time_xgboost_norm - start_time_xgboost_norm
hybrid_execution_time_2 = end_time_hybrid_norm - start_time_hybrid_norm

total_execution_time_2 = end_time_program_2 - start_time_program_2

print(f"ARIMA Execution Time: {arima_execution_time_2:.4f} seconds")
print(f"XGBoost Execution Time: {xgboost_execution_time_2:.4f} seconds")
print(f"Hybrid Model Execution Time: {hybrid_execution_time_2:.4f} seconds")
print(f"Total Program Execution Time: {total_execution_time_2:.4f} seconds")

import matplotlib.pyplot as plt
import pandas as pd

# Data for Program 1 (HPC version)
program_1_results = {
    "Model": ["ARIMA", "XGBoost", "Hybrid", "Total Program"],
    "RMSE": [arima_rmse, xgboost_rmse, hybrid_rmse, None],
    "Execution Time (s)": [arima_execution_time, xgboost_execution_time, hybrid_execution_time, total_execution_time]
}

# Data for Program 2 (Non-HPC version)
program_2_results = {
    "Model": ["ARIMA", "XGBoost", "Hybrid", "Total Program"],
    "RMSE": [arima_rmse_2, xgboost_rmse_2, hybrid_rmse_2, None],
    "Execution Time (s)": [arima_execution_time_2, xgboost_execution_time_2, hybrid_execution_time_2, total_execution_time_2]
}

# Create DataFrames
df_program_1 = pd.DataFrame(program_1_results)
df_program_1['Program'] = 'Program 1 (HPC)'

df_program_2 = pd.DataFrame(program_2_results)
df_program_2['Program'] = 'Program 2 (Non-HPC)'

# Concatenate both DataFrames
df_comparison = pd.concat([df_program_1, df_program_2], ignore_index=True)

# Print the comparison DataFrame
print("\nModel Performance Comparison:")
print(df_comparison)

# Plot RMSE Comparison
plt.figure(figsize=(10, 6))
for model in df_comparison['Model'].unique():
    subset = df_comparison[df_comparison['Model'] == model]
    plt.bar(subset['Program'] + ' ' + subset['Model'], subset['RMSE'], label=model, width=0.4, align='center')

plt.title('RMSE Comparison Between Program 1 (HPC) and Program 2 (Non-HPC)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Execution Time Comparison
plt.figure(figsize=(10, 6))
for model in df_comparison['Model'].unique():
    subset = df_comparison[df_comparison['Model'] == model]
    plt.bar(subset['Program'] + ' ' + subset['Model'], subset['Execution Time (s)'], label=model, width=0.4, align='center')

plt.title('Execution Time Comparison Between Program 1 (HPC) and Program 2 (Non-HPC)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()