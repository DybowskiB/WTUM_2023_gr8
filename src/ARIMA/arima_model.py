import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('../data/preprocessed_train_data.csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

data = data[(data['store_nbr'] == 1) & (data['family'] == 10)]
data.set_index('date', inplace=True)
data.index.freq = 'D'

# Split your data into training and testing sets
n_train = int(0.8 * len(data))
train_data = data.iloc[:n_train]
test_data = data.iloc[n_train:]

# Define the order of the ARIMA model
p = 1
d = 1
q = 1

# Fit the ARIMA model to the training data
model = ARIMA(endog=train_data['sales'],
              exog=train_data[['onpromotion', 'holiday_event_type', 'locale', 'locale_name', 'description',
                               'transferred', 'dcoilwtico', 'city', 'state', 'store_type', 'cluster',
                               'transactions', 'paid_day']],
              order=(p, d, q))
results = model.fit()

# Make forecasts for future time periods
predictions = results.predict(start=len(train_data), end=len(data) - 1,
                              exog=test_data[['onpromotion', 'holiday_event_type', 'locale', 'locale_name',
                                              'description', 'transferred', 'dcoilwtico', 'city', 'state',
                                              'store_type', 'cluster', 'transactions', 'paid_day']],
                              dynamic=False)

# Calculate mean squared error of predictions
mse = mean_squared_error(test_data['sales'], predictions)
print('MSE:', mse)

# Plot actual values and predicted values
plt.plot(train_data.index, train_data['sales'], label='train data')
plt.plot(test_data.index, test_data['sales'], label='test data')
plt.plot(predictions.index, predictions, label='predicted values')
plt.legend()
plt.show()
