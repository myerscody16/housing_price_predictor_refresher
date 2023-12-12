import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


class housing_price_predictor:
    def __init__(self):
        self.data = pd.read_csv('data/processed_data.csv')


    def split_data(self):
        training_ratio = .8
        test_ratio = .15
        validation_ratio = .05
        X = self.data.drop(['SalePrice'], axis=1)
        Y = self.data['SalePrice']
        self.X_training, self.X_testing, self.Y_training, self.Y_testing = train_test_split(X, Y, test_size=1-training_ratio)


    def train(self):
        self.model_svr = svm.SVR()
        self.model_svr.fit(self.X_training, self.Y_training)
        self.model_rfr = RandomForestRegressor(n_estimators=10)
        self.model_rfr.fit(self.X_training, self.Y_training)


    def predict(self):
        Y_pred = self.model_svr.predict(self.X_testing)
        print(f'support vector machine model : \n{mean_absolute_percentage_error(self.Y_testing, Y_pred)=}')
        Y_pred = self.model_rfr.predict(self.X_testing)
        print(f'random forest model : \n{mean_absolute_percentage_error(self.Y_testing, Y_pred)=}')


