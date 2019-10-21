from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class Support_vector_regressor:
    svr = None

    def __init__(self, c=1.0, k='rbf', g='auto', e=0.1):
        self.svr = SVR(C=c, kernel=k, gamma=g, epsilon=e)

    def train(self, x_train=None, y_train=None):
        try:
            self.svr.fit(x_train, y_train)
        except:
            print("Support_vector_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.svr.predict(x_test), y_test)
        except:
            print("Support_vector_regressor: x_test or y_test may be wrong")
