import pandas as pd
from sklearn.preprocessing import LabelEncoder


def check_missing_values(data):
    """
    Checks if data has missing values, if it has then throw error
    :return:
    """
    missing_values_count = data.isnull().sum()
    is_zero = missing_values_count.eq(0).all()
    if not is_zero:
        raise ValueError("Data is not complete")


def time_weighted_interpolation(data, column):
    """
    Time-weighted oil price interpolation
    :return: interpolated data
    """
    oil_data_date = data['date']
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data[column].interpolate(method='time', inplace=True, limit_direction='both')
    data.reset_index(inplace=True)
    data['date'] = oil_data_date
    return data


def merge_data(data):
    """
    Merge data
    """
    # Merge holidays events data
    data = pd.merge(data, holidays_events_data, on='date', how='left')
    # Merge oil data
    data = pd.merge(data, oil_data, on='date', how='left')
    # Merge stores data
    data = pd.merge(data, stores_data, on='store_nbr', how='left')
    # Merge transactions data
    data = pd.merge(data, transactions_data, on=['date', 'store_nbr'], how='left')
    # Perform time-weighted interpolation to gain oil data
    data = time_weighted_interpolation(data, 'dcoilwtico')
    # Interpolate missing transactions values
    data = time_weighted_interpolation(data, 'transactions')

    # Fill missing values connected with holidays events with a default value
    data['holiday_event_type'].fillna("None", inplace=True)
    data['locale'].fillna("None", inplace=True)
    data['locale_name'].fillna("None", inplace=True)
    data['description'].fillna("None", inplace=True)
    data['transferred'].fillna("None", inplace=True)

    return data


def convert_string_to_int(string_val):
    """
    Convert string value to int
    :param string_val: string value
    :return: number_values - converted values
    """
    le = LabelEncoder()
    number_values = le.fit_transform(string_val)
    return number_values


def convert_all_values(data):
    """
    Function converts provided string values in columns to int
    :param data: Data to convert
    :return: Converted all string columns from data
    """
    data['family'] = convert_string_to_int(data['family'])
    data['holiday_event_type'] = convert_string_to_int(data['holiday_event_type'])
    data['locale'] = convert_string_to_int(data['locale'])
    data['locale_name'] = convert_string_to_int(data['locale_name'])
    data['description'] = convert_string_to_int(data['description'])
    data['transferred'] = [str(d) for d in data['transferred']]
    data['transferred'] = convert_string_to_int(data['transferred'])
    data['city'] = convert_string_to_int(data['city'])
    data['state'] = convert_string_to_int(data['state'])
    data['store_type'] = convert_string_to_int(data['store_type'])
    data['paid_day'] = convert_string_to_int(data['paid_day'])
    return data


def add_data_about_paid_days(data):
    """
    Specify if day is paid
    :param data: Data with 'date' column
    :return:
    """
    data['paid_day'] = False
    data['date'] = pd.to_datetime(data['date'])
    data.loc[(data['date'].dt.day == 15) | data['date'].dt.is_month_end, 'paid_day'] = True
    data['date'] = [d.strftime('%Y-%m-%d') for d in data['date']]


def preprocess(data, filename):
    """
    Full data preprocessing
    :param data: Data to preprocess
    :param filename: File in which data will be saved
    :return:
    """
    # Merge train data
    data = merge_data(data)
    # Add data about paid days in public sector
    add_data_about_paid_days(data)
    # Convert string values to number (int)
    data = convert_all_values(data)
    # Check train_data correctness
    try:
        check_missing_values(data)
    except ValueError:
        missing_values_count = data.isnull().sum()
        print("Preprocessing error (" + filename + ").\n")
        print("Missing values count:\n", missing_values_count)
    else:
        # save to file
        data = data.drop_duplicates(subset=['date', 'store_nbr', 'family'])
        data.to_csv('data/' + filename, index=False)
        print("\nPreprocessing completed.\nPreprocessed data has been saved to " + filename)


def add_missing_date(data):
    """
    Add missing date values
    :param data: Data in which missing dates will be added
    :return: Complete data
    """
    global dates
    start_date = min(data['date'])
    end_date = max(data['date'])
    dates = pd.date_range(start=start_date, end=end_date)
    stores = data['store_nbr'].unique()
    families = data['family'].unique()
    complete_data = []
    for d in dates:
        for s in stores:
            for f in families:
                row = (d, s, f)
                complete_data.append(row)
    complete_data = pd.DataFrame(complete_data, columns=['date', 'store_nbr', 'family'])
    complete_data['date'] = [d.strftime('%Y-%m-%d') for d in complete_data['date']]
    data = pd.merge(complete_data, data.drop('id', axis=1),
                    on=['date', 'store_nbr', 'family'], how='left')

    # time weighted interpolation of missing sales and onpromotion values
    if 'sales' in data.columns:
        data = time_weighted_interpolation(data, 'sales')
    data = time_weighted_interpolation(data, 'onpromotion')

    return data


'''
Preprocess oil data
'''

# Read oil data
oil_data = pd.read_csv('data/oil.csv')

# Perform time-weighted oil price interpolation
oil_data = time_weighted_interpolation(oil_data, 'dcoilwtico')

# Check data correctness
check_missing_values(oil_data)


'''
Preprocess holiday events data
'''

# Read holiday events data
holidays_events_data = pd.read_csv('data/holidays_events.csv')
holidays_events_data.rename(columns={'type': 'holiday_event_type'}, inplace=True)

# Check data correctness
check_missing_values(holidays_events_data)


'''
Preprocess stores data
'''

# Read stores data
stores_data = pd.read_csv('data/stores.csv')
stores_data.rename(columns={'type': 'store_type'}, inplace=True)

# Check data correctness
check_missing_values(stores_data)


'''
Preprocess transactions data
'''

# Read transactions data
transactions_data = pd.read_csv('data/transactions.csv')

# Add missing dates
dates = pd.date_range(start='2017-08-16', end='2017-08-31', normalize=True)
dates = [d.strftime('%Y-%m-%d') for d in dates]
store_nbr = transactions_data['store_nbr'].unique()
for date in dates:
    for store in store_nbr:
        transactions_data = transactions_data._append({'date': date, 'store_nbr': store}, ignore_index=True)

# Interpolate missing transactions values based on date and store_nbr
transactions_data['transactions'] = transactions_data['transactions'].interpolate(method='linear',
                                                                                  limit_direction='both',
                                                                                  column=['date', 'store_nbr'])

# check data correctness
check_missing_values(transactions_data)


'''
Preprocess train and test data
'''

# Train data
train_data = pd.read_csv('data/train.csv')
train_data = add_missing_date(train_data)
preprocess(train_data, 'preprocessed_train_data.csv')

# Test data
test_data = pd.read_csv('data/test.csv')
test_data = add_missing_date(test_data)
preprocess(test_data, 'preprocessed_test_data.csv')
