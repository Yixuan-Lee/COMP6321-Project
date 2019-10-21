import os
import numpy as np
from models import settings
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_regression import Linear_regression
from neural_network_regressor import Neural_network_regressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

class Wine_quality:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/1_Wine_Quality'
        filename1 = 'winequality-red.csv'
        filename2 = 'winequality-white.csv'

        f1 = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename1),
            delimiter=';', dtype=np.float32, skiprows=1)
        f2 = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename2),
            delimiter=';', dtype=np.float32, skiprows=1)
        f = np.vstack((f1, f2))
        self.data = f[:, :-1]
        self.targets = f[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

    def support_vector_regression(self):
        svr = Support_vector_regressor()
        svr.train(self.x_train, self.y_train)
        return svr.get_mse(self.x_test, self.y_test)

    def decision_tree_regression(self):
        dtf = Decision_tree_regressor()
        dtf.train(self.x_train, self.y_train)
        return dtf.get_mse(self.x_test, self.y_test)

    def random_forest_regression(self):
        rfr = Random_forest_regressor()
        rfr.train(self.x_train, self.y_train)
        return rfr.get_mse(self.x_test, self.y_test)

    def ada_boost_regression(self):
        abr = Ada_boost_regressor()
        abr.train(self.x_train, self.y_train)
        return abr.get_mse(self.x_test, self.y_test)

    def gaussian_process_regression(self):
        kernel = DotProduct() + WhiteKernel()
        gpr = Gaussian_process_regressor(k=kernel)
        gpr.train(self.x_train, self.y_train)
        return gpr.get_mse(self.x_test, self.y_test)

    def linear_regression(self):
        lr = Linear_regression()
        lr.train(self.x_train, self.y_train)
        return lr.get_mse(self.x_test, self.y_test)

    def neural_network_regression(self):
        nnr = Neural_network_regressor(hls=(15,), s='lbfgs')
        nnr.train(self.x_train, self.y_train)
        return nnr.get_mse(self.x_test, self.y_test)


if __name__ == '__main__':
    wq = Wine_quality()
    print('SVR: %.5f' % wq.support_vector_regression())
    print('DTF: %.5f' % wq.decision_tree_regression())
    print('RFR: %.5f' % wq.random_forest_regression())
    print('ABR: %.5f' % wq.ada_boost_regression())
    print('GPR: %.5f' % wq.gaussian_process_regression())
    print(' LR: %.5f' % wq.linear_regression())
    print('NNR: %.5f' % wq.neural_network_regression())


