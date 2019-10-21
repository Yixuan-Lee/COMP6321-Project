from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


class Neural_network_regressor:
    nnr = None

    def __init__(self, hls=(100,), a='relu', s='adam', alp=1e-4, rs=0):
        self.nnr = MLPRegressor(hidden_layer_sizes=hls, activation=a,
            solver=s, alpha=alp, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.nnr.fit(x_train, y_train)
        except:
            print("Neural_network_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.nnr.predict(x_test), y_test)
        except:
            print("Neural_network_regressor: x_test or y_test may be wrong")
