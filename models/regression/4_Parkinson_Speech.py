import os
import scipy
import numpy as np
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_regression import Linear_regression
from neural_network_regressor import Neural_network_regressor


class Parkinson_speech:
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/4_Parkinson_Speech'
        filename_train = 'train_data.txt'
        filename_test = 'test_data.txt'

        train_data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
            filename_train), delimiter=',')
        test_data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
            filename_test), delimiter=',')
        self.x_train = train_data[:, 1:27]
        self.y_train = train_data[:, -1]
        self.x_test = test_data[:, 1:-1]
        self.y_test = test_data[:, -1]

    def support_vector_regression(self):
        kernel = ('sigmoid', 'rbf')
        C = scipy.stats.reciprocal(1, 100)
        gamma = scipy.stats.reciprocal(0.01, 20)
        coef0 = scipy.stats.uniform(0, 5)
        epsilon = scipy.stats.reciprocal(0.01, 1)

        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            epsilon=epsilon,
            random_search=True)

        svr.print_parameter_candidates()
        svr.print_best_estimator()

        return svr.mean_sqaured_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_regression(self):
        max_depth = range(1, 10)
        min_samples_leaf = range(1, 9)

        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=50,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_search=True)

        dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return dtr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_regression(self):
        n_estimators = range(1, 60)
        max_depth = range(1, 20)

        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=50,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_search=True)

        rfr.print_parameter_candidates()
        rfr.print_best_estimator()

        return rfr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def ada_boost_regression(self):
        n_estimators = range(1, 100)

        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=99,
            n_estimators=n_estimators,
            random_search=True)

        abr.print_parameter_candidates()
        abr.print_best_estimator()

        return abr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_process_regression(self):
        pass

    def linear_regression(self):
        lr = Linear_regression(
            x_train=self.x_train,
            y_train=self.y_train)

        return lr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def neural_network_regression(self):
        pass


if __name__ == '__main__':
    ps = Parkinson_speech()
    print("mean squared error on the actual test set:")
    print('SVR: %.5f' % (ps.support_vector_regression()))
    print('DTR: %.5f' % (ps.decision_tree_regression()))
    print('RFR: %.5f' % (ps.random_forest_regression()))
    print('ABR: %.5f' % (ps.ada_boost_regression()))
    print('GPR: %.5f' % (ps.gaussian_process_regression()))
    print(' LR: %.5f' % (ps.linear_regression()))
    print('NNR: %.5f' % (ps.neural_network_regression()))

