from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class Random_forest_regressor:
    rfr = None

    def __init__(self, n=10, c='mse', md=None, rs=0):
        self.rfr = RandomForestRegressor(n_estimators=n, criterion=c,
            max_depth=md, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.rfr.fit(x_train, y_train)
        except:
            print("Random_forest_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.rfr.predict(x_test), y_test)
        except:
            print("Random_forest_regressor: x_test or y_test may be wrong")
