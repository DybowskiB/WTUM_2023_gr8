import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import sys


def prepare_data(store_nbr, family):

    # Load data
    data = pd.read_csv('../data/preprocessed_train_data.csv')
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    data = data[(data['store_nbr'] == store_nbr) & (data['family'] == family)]
    data.set_index('date', inplace=True)
    data.index.freq = 'D'

    return data


def train(data):

    # Print info about training start
    print("Start training process ...")

    # Split data into training and testing sets
    n_train = int(0.8 * len(data))
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]

    # Declare features
    exog_train_data = train_data[['onpromotion', 'holiday_event_type', 'locale', 'locale_name',
                                  'description', 'transferred', 'dcoilwtico', 'transactions', 'paid_day']]
    exog_test_data = test_data[['onpromotion', 'holiday_event_type', 'locale', 'locale_name',
                                'description', 'transferred', 'dcoilwtico', 'transactions', 'paid_day']]

    # Get the optimal order of the ARIMA model
    p_optimal = d_optimal = q_optimal = 0
    p_max = 11
    d_max = 4
    q_max = 6
    p_range = range(1, p_max)
    d_range = range(0, d_max)
    q_range = range(1, q_max)
    best_predictions = []
    min_mse = sys.maxsize
    iteration = 0
    max_iteration = (p_max - 1) * d_max * (q_max - 1)
    for p in p_range:
        for d in d_range:
            for q in q_range:

                # Print info about progress
                print('Iteration: ', iteration, " / ", max_iteration)
                iteration = iteration + 1

                # Fit the ARIMA model to the training data
                model = ARIMA(endog=train_data['sales'], exog=exog_train_data, order=(p, d, q))
                results = model.fit(method_kwargs={'maxiter': 1000})

                # Make forecasts for future time periods
                predictions = results.predict(start=len(train_data), end=len(data) - 1,
                                              exog=exog_test_data, dynamic=False)
                predictions.loc[predictions < 0] = 0

                # Check if it is the best model
                mse = mean_squared_error(test_data['sales'], predictions)
                if mse < min_mse:
                    p_optimal = p
                    d_optimal = d
                    q_optimal = q
                    best_predictions = predictions

    return test_data, best_predictions, p_optimal, d_optimal, q_optimal


def calculate_mse_plot(test_data, predictions, draw):

    # Calculate mean squared error of predictions
    mse = mean_squared_error(test_data['sales'], predictions)
    if draw:
        print('MSE:', mse)

    # Plot actual values and predicted values
    if draw:
        plt.plot(test_data.index, test_data['sales'], label='test data')
        plt.plot(predictions.index, predictions, label='predicted values')
        plt.legend()
        plt.show()

    return mse
