import xgboost as xgb
import pandas as pd
from xgboost_model import XGBModel
from sklearn.model_selection import train_test_split
import numpy as np
import os


DATA_PATH = "src/data/preprocessed_train_data.csv"


def load_data():
    train_data = pd.read_csv(DATA_PATH, index_col=[0])
    train_data, test_data = train_test_split(train_data)
    X_train = train_data.drop(["sales"], axis=1)
    y_train = train_data["sales"]
    X_test = test_data.drop(["sales"], axis=1)
    y_test = test_data['sales']
    return X_train, y_train, X_test, y_test
    

def main():
    X_train, y_train, X_test, y_test = load_data()
    model = XGBModel()
    model.fit(X_train, y_train)
  
  
if __name__ == "__main__":
    main()