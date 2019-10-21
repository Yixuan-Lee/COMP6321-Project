from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error


class Gaussian_process_regressor:
    gpr = None

    def __init__(self, k=RBF(), alp=1e-10, nro=0, rs=0):
        self.gpr = GaussianProcessRegressor(kernel=k, alpha=alp,
            n_restarts_optimizer=nro, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.gpr.fit(x_train, y_train)
        except:
            print("Gaussian_process_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.gpr.predict(x_test), y_test)
        except:
            print("Gaussian_process_regressor: x_test or y_test may be wrong")
