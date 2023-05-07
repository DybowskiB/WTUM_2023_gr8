import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

class XGBModel:
    def __init__(self, num_round=10, max_depth=3, learning_rate=0.1, min_child_weight=1):
        self.num_round = num_round
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.model = None
    
    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        param = {'max_depth': self.max_depth,
                 'min_child_weight': self.min_child_weight,
                 'eta': self.learning_rate, 
                 "objective": "reg:squarederror"}
        self.model = xgb.train(param, dtrain, self.num_round)
    
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        y_pred = self.model.predict(dtest)
        return y_pred
    
    def evaluate(self, X_test, y_test):
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        return {'mse': mse, 'rmse': rmse, 'r2': r2, 'evs': evs}
