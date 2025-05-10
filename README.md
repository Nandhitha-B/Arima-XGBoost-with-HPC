# Arima-XGBoost-with-HPC
Integration of High- Performance Computing (HPC) techniques with a hybrid ARIMA-XGBoost model to optimize time series forecasting tasks.

Execution Procedure for the Stock Price Prediction Model
This document provides the execution procedure for the stock price prediction model implemented using various machine learning models such as ARIMA, XGBoost, and a hybrid model that combines the predictions of ARIMA and XGBoost. The model is trained on Apple stock data retrieved from Yahoo Finance and utilizes Dask for parallel computation to improve performance.
1. Install Required Libraries
Ensure you have the following libraries installed using the following pip command:

```bash
pip install numpy pandas statsmodels joblib dask distributed xgboost dask-ml yfinance matplotlib
pip install pytorch_lightning
pip install u8darts
pip install darts
pip install arch
pip install tensorflow


```
2. Setup and Initialize Dask Client
Dask is used to distribute computations across multiple processes or workers. Initialize a Dask client at the beginning of the script:

```python
client = Client()
```
3. Download Data from Yahoo Finance
The model uses Apple's stock data (`AAPL`) from Yahoo Finance, retrieved from January 1, 2020, to January 1, 2023. The data is then processed to only include the 'Close' price.

```python
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data = data[['Close']]
data.index = pd.to_datetime(data.index)
data = data.asfreq('B').ffill()
```
4. Preprocess Data
The model computes technical indicators based on the closing price, such as returns, moving averages (5-day, 20-day), and lag features. Missing values are dropped from the resulting dataset.

```python
indicators['Return'] = data['Close'].pct_change()
indicators['MA_5'] = data['Close'].rolling(window=5).mean()
indicators['MA_20'] = data['Close'].rolling(window=20).mean()
indicators['Lag_1'] = data['Close'].shift(1)
indicators = indicators.dropna()
indicators = dd.from_pandas(indicators, npartitions=4)
```
5. Split Data into Train and Test Sets
The data is split into training and testing sets, with 80% of the data used for training and the remaining 20% for testing.

```python
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
```
6. Fit ARIMA Model
An ARIMA model is fit to the training data. The model is used to generate predictions for the stock prices. The predictions are added to the indicators DataFrame.

```python
model = ARIMA(train_data, order=(1, 0, 1))
model_fit = model.fit()
indicators['ARIMA_Pred'] = model_fit.predict(start=data.index[0], end=data.index[-1])
```
7. Fit XGBoost Model
The XGBoost model is trained on the same features used for ARIMA predictions. Dask's `DaskXGBRegressor` is used for parallel training.

```python
dask_model = xgb.dask.DaskXGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
dask_model.fit(X_train_dask, y_train_dask)
```
8. Create Hybrid Model with Stacking
The hybrid model combines the predictions of the ARIMA and XGBoost models using stacking. A Linear Regression model is trained on the predictions of ARIMA and XGBoost.

```python
stacking_model = LinearRegression()
stacking_model.fit(X_stack, y_stack)
indicators_test_aligned['Hybrid_Pred'] = stacking_model.predict(X_stack)
```
9. Evaluate Performance
The models' performances are evaluated using the RMSE (Root Mean Squared Error) between the actual and predicted values for the test set.

```python
rmse_arima = np.sqrt(mean_squared_error(test['Close'], indicators_test_aligned['ARIMA_Pred']))
rmse_xgb = np.sqrt(mean_squared_error(test['Close'], indicators_test_aligned['XGBoost_Pred']))
rmse_hybrid = np.sqrt(mean_squared_error(test['Close'], indicators_test_aligned['Hybrid_Pred']))
```
10. Measure Execution Time
The execution time of the models is measured for both the configurations with and without HPC (High Performance Computing) using the `measure_execution_time` function.

```python
time_without_hpc = measure_execution_time(without_hpc_execution, 'Without HPC')
time_with_hpc = measure_execution_time(with_hpc_execution, 'With HPC')
```
11. Display Results and Performance Summary
The performance results are summarized in a DataFrame, and visualized in bar charts comparing execution time and RMSE values.

```python
summary_df = pd.DataFrame(results)
summary_df['Execution_Time'] = [time_without_hpc, time_with_hpc]
```
