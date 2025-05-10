

from darts.utils.missing_values import fill_missing_values
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 1: Download AAPL stock data
ticker = "AAPL"
data = yf.download(ticker, start="2024-01-01", end="2025-01-01")

# Step 2: Use the Adjusted Close Price for forecasting
data = data[['Close']]

# Step 3: Remove NaN values from the data
data = data.dropna()

# Step 4: Normalize the data to make training more efficient
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Split the data into training and validation sets (80-20 split)
train_size = int(len(data) * 0.8)
train_data, val_data = scaled_data[:train_size], scaled_data[train_size:]

# Step 6: Convert the training and validation sets to Darts TimeSeries objects
train_series = TimeSeries.from_times_and_values(pd.to_datetime(
    data.index[:train_size]), train_data.flatten(), fill_missing_dates=True, freq='1D')
val_series = TimeSeries.from_times_and_values(pd.to_datetime(
    data.index[train_size:]), val_data.flatten(), fill_missing_dates=True, freq='1D')

# Step 7: Check for NaN values in TimeSeries
print("NaN values in train_series:", np.isnan(train_series.values()).sum())
print("NaN values in val_series:", np.isnan(val_series.values()).sum())

# Step 8: Handle NaN values (if any)

train_series = fill_missing_values(train_series)
val_series = fill_missing_values(val_series)

# Step 9: Initialize and train the TiDE model
model = TiDEModel(
    input_chunk_length=60,     # Look at the last 60 days
    output_chunk_length=10,    # Predict the next 10 days
    num_encoder_layers=2,      # Encoder with 2 layers
    num_decoder_layers=2,      # Decoder with 2 layers
    hidden_size=128,           # Hidden layer size
    dropout=0.1,               # Dropout rate
    use_static_covariates=False,  # No static covariates in this case
    random_state=42            # Set the random_state for reproducibility
)

# Train the model on the training data
model.fit(train_series)

# Step 10: Forecast the next 10 days (validation period)
forecast = model.predict(n=10)

# Step 11: Check for NaN values in forecast
print("NaN values in forecast:", np.isnan(forecast.values()).sum())

# Step 12: Handle NaN values in forecast (if any)
forecast = fill_missing_values(forecast)

# Step 13: Invert scaling to get original price values
forecast_values = scaler.inverse_transform(forecast.values().reshape(-1, 1))
actual_values = scaler.inverse_transform(val_data[:10].reshape(-1, 1))

# Step 14: Plot the forecasted values and actual values
plt.figure(figsize=(10, 6))
plt.plot(data.index[train_size:train_size+10],
         forecast_values, label='Predicted', color='orange')
plt.plot(data.index[train_size:train_size+10],
         actual_values, label='Actual', color='blue')
plt.legend()
plt.title("AAPL Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Step 15: Evaluate the model using MAPE, RMSE, R², MAE, MSE
rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
r2 = r2_score(actual_values, forecast_values)
mae = mean_absolute_error(actual_values, forecast_values)
mse = mean_squared_error(actual_values, forecast_values)
error = mape(TimeSeries.from_times_and_values(pd.to_datetime(data.index[train_size:train_size+10]), actual_values.flatten()),
             TimeSeries.from_times_and_values(pd.to_datetime(data.index[train_size:train_size+10]), forecast_values.flatten()))

# Step 16: Print the performance metrics in a table
metrics = pd.DataFrame({
    'Metric': ['MAPE', 'RMSE', 'R²', 'MAE', 'MSE'],
    'Value': [error, rmse, r2, mae, mse]
})

print(metrics)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

# Load Data
data = yf.download("AAPL", start="2024-01-01", end="2025-01-01")
data = data[['Close']]

# Store calculated columns in a separate DataFrame
indicators = pd.DataFrame()

# Calculate technical indicators
indicators['Return'] = data['Close'].pct_change()
indicators['MA_5'] = data['Close'].rolling(window=5).mean()
indicators['MA_20'] = data['Close'].rolling(window=20).mean()
indicators['Lag_1'] = data['Close'].shift(1)

data.index = pd.to_datetime(data.index)
data = data.asfreq('B').ffill()

# Split Data into Training and Testing Sets
train_size = int(len(data) * 0.8)  # Use 80% of the data for training
train, test = data[:train_size], data[train_size:]

# Fit SARIMA Model (tuned order)
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))  # Adjust parameters based on data insights
model_fit = model.fit(disp=False)

# Predict for the entire dataset and ensure it's a Series
indicators['SARIMA_Pred'] = model_fit.predict(start=data.index[0], end=data.index[-1])
indicators['Close'] = data['Close']
indicators.index = pd.to_datetime(indicators.index)
indicators = indicators.asfreq('B').ffill()

# Drop missing values (due to shifting and rolling)
indicators = indicators.dropna()

# Align target variable with the indicator's index
y = data['Close'].loc[indicators.index]

# Prepare data for XGBoost
X = indicators[['Return', 'MA_5', 'MA_20', 'Lag_1', 'SARIMA_Pred']]

# Split data into train and test sets
train_size = int(len(X) * 0.8)  # 80% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define XGBoost model with improved parameters
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,  # Adjust learning rate for better results
    n_estimators=200,    # Number of trees
    max_depth=5,         # Control overfitting by limiting depth
    min_child_weight=3,  # Minimum sum of instance weight
    subsample=0.8,       # Randomly sample the training data
    colsample_bytree=0.8, # Subsample features
    gamma=0.1            # Regularization parameter to reduce overfitting
)

# Train the XGBoost model
model_xgb.fit(X_train, y_train)

# Predict using the trained XGBoost model
y_pred_xgb = model_xgb.predict(X_test)

# Calculate RMSE (Root Mean Squared Error) for XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# Ensure the test indices are aligned correctly with the features
indicators_test_aligned = indicators.loc[test.index]

# Align X_test with the same indices as indicators_test_aligned
X_test_aligned = X.loc[test.index]

# Predict using the XGBoost model for the aligned X_test
indicators_test_aligned['XGBoost_Pred'] = model_xgb.predict(X_test_aligned)

# Now you can safely proceed with the rest of the code
# Continue with the previous steps for combining the predictions, etc.

# Include polynomial features to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(indicators_test_aligned[['SARIMA_Pred', 'XGBoost_Pred']])

# Initialize Ridge Regression (L2 regularization)
ridge_model = Ridge()

# Hyperparameter tuning using GridSearchCV
param_grid = {'alpha': [0.1, 1, 10, 100, 200]}  # Grid for alpha
grid_search = GridSearchCV(ridge_model, param_grid, cv=5)  # 5-fold cross-validation

# Fit the model using GridSearchCV to find the best alpha
grid_search.fit(X_poly, indicators_test_aligned['Close'])

# Get the best model from grid search
best_ridge_model = grid_search.best_estimator_

# Predict using the best Ridge model
indicators_test_aligned['Hybrid_Pred'] = best_ridge_model.predict(X_poly)

# Calculate RMSE, MSE, and R-squared for Hybrid
rmse_hybrid = np.sqrt(mean_squared_error(indicators_test_aligned['Close'], indicators_test_aligned['Hybrid_Pred']))
mse_hybrid = mean_squared_error(indicators_test_aligned['Close'], indicators_test_aligned['Hybrid_Pred'])
r2_hybrid = r2_score(indicators_test_aligned['Close'], indicators_test_aligned['Hybrid_Pred'])

# Calculate RMSE, MSE, and R-squared for XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Calculate RMSE, MSE, and R-squared for SARIMA
rmse_sarima = np.sqrt(mean_squared_error(indicators_test_aligned['Close'], indicators_test_aligned['SARIMA_Pred']))
mse_sarima = mean_squared_error(indicators_test_aligned['Close'], indicators_test_aligned['SARIMA_Pred'])
r2_sarima = r2_score(indicators_test_aligned['Close'], indicators_test_aligned['SARIMA_Pred'])

# Print results for all models
print(f"RMSE for Hybrid: {rmse_hybrid:.4f}, MSE: {mse_hybrid:.4f}, R-squared: {r2_hybrid:.4f}")
print(f"RMSE for XGBoost: {rmse_xgb:.4f}, MSE: {mse_xgb:.4f}, R-squared: {r2_xgb:.4f}")
print(f"RMSE for SARIMA: {rmse_sarima:.4f}, MSE: {mse_sarima:.4f}, R-squared: {r2_sarima:.4f}")


# Plotting the predictions
plt.figure(figsize=(14, 8))

# Plot the actual Close prices
plt.plot(indicators_test_aligned.index, indicators_test_aligned['Close'], label='Actual Close', color='black', linewidth=2)

# Plot SARIMA predictions
plt.plot(indicators_test_aligned.index, indicators_test_aligned['SARIMA_Pred'], label='SARIMA Predictions', linestyle='--', color='blue')

# Plot XGBoost predictions
plt.plot(indicators_test_aligned.index, indicators_test_aligned['XGBoost_Pred'], label='XGBoost Predictions', linestyle='--', color='red')

# Plot Hybrid predictions
plt.plot(indicators_test_aligned.index, indicators_test_aligned['Hybrid_Pred'], label='Hybrid Predictions', linestyle='-', color='green')

# Add titles and labels
plt.title('Comparison of Model Predictions vs Actual Close Prices', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)

# Add legend
plt.legend()

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import yfinance as yf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# Download AAPL stock data from Yahoo Finance
data = yf.download('AAPL', start='2024-01-01', end='2025-01-01')
data = data[['Close']]  # Use only 'Close' price

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate error metrics
actual_values = np.concatenate((trainY[0], testY[0]), axis=0)
predicted_values = np.concatenate((trainPredict[:,0], testPredict[:,0]), axis=0)

# Calculate R², RMSE, MSE, MAE
lr2 = r2_score(actual_values, predicted_values)
lrmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
lmse = mean_squared_error(actual_values, predicted_values)

# Print error metrics
print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
print(f"MSE: {mse}")

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Plot actual vs predicted values
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(dataset), label="Actual AAPL Close Prices")
plt.plot(trainPredictPlot, label="Train Prediction")
plt.plot(testPredictPlot, label="Test Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("AAPL Stock Price Prediction using LSTM")
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Define the metrics for each model
metrics_data = {
    'Model': ['TiDE', 'ARIMA', 'XGBoost', 'Hybrid', 'LSTM'],
    'RMSE': [rmse, rmse_sarima, rmse_xgb, rmse_hybrid, lrmse],
    'R²': [r2, r2_sarima, r2_xgb,r2_hybrid, lr2],
    'MSE': [mse, mse_sarima, mse_xgb, mse_hybrid, lmse]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
print(metrics_df)

# Plot the table
fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the figure size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 loc='center',
                 cellLoc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)  # Adjust table scale

# Add a title
plt.title("Model Comparison: RMSE, R², and MSE", fontsize=14, pad=20)

# Show the plot
plt.show()

import numpy as np

# Values for the radar chart
metrics = ['RMSE', 'R²', 'MSE']
models = metrics_df['Model']
values = metrics_df[['RMSE', 'R²', 'MSE']].values.T  # Transpose for plotting

# Number of metrics
num_metrics = len(metrics)

# Set up the radar chart
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
values = np.concatenate((values, values[:,[0]]), axis=1)  # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, model in enumerate(models):
    ax.plot(angles, values[:, i], label=model, linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles)
ax.set_xticklabels(metrics)

plt.title("Radar Chart of RMSE, R², and MSE for Each Model", fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.show()