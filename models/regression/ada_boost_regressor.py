from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


class Ada_boost_regressor:
    abr = None

    def __init__(self, n=50, lr=1, l='linear', rs=0):
        self.abr = AdaBoostRegressor(n_estimators=n, learning_rate=lr, loss=l,
            random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.abr.fit(x_train, y_train)
        except:
            print("Ada_boost_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.abr.predict(x_test), y_test)
        except:
            print("Ada_boost_regressor: x_test or y_test may be wrong")
