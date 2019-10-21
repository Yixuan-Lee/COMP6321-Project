from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class Linear_regression:
    lr = None

    def __init__(self, n=False, nj=None):
        self.lr = LinearRegression(normalize=n, n_jobs=nj)

    def train(self, x_train=None, y_train=None):
        try:
            self.lr.fit(x_train, y_train)
        except:
            print("Linear_regression: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.lr.predict(x_test), y_test)
        except:
            print("Linear_regression: x_test or y_test may be wrong")
