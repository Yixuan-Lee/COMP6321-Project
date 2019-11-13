import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from scipy.stats import norm    # For tuning parameters
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_least_squares import Linear_least_squares
from neural_network_regressor import Neural_network_regressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

        # read the data file
        f1 = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename1),
            delimiter=';', dtype=np.float32, skiprows=1)
        f2 = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename2),
            delimiter=';', dtype=np.float32, skiprows=1)
        file_data = np.vstack((f1, f2))
        self.data = file_data[:, :-1]  # (6497, 11)
        self.targets = file_data[:, -1]  # (6497, )

        # subsampling data (for speeding up)
#         self.data = self.data[:1000]
#         self.targets = self.targets[:1000]

        # split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set and testing set
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    ##################### Model training #####################
    def support_vector_regression(self):
        """
        for svr, i train on the training data using different :
            1) C
            2) gamma
            3) kernel
        :return: test accuracy of the svr best model
        """
        # define parameters
#         C = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)
#         gamma = np.logspace(start=-4, stop=0, base=10, num=5, dtype=np.float32)
#         kernel = ('rbf', 'linear', 'sigmoid')
        # best result over C: 1.0
        # best result over gamma: 0.1
        # best result over kernel: 'rbf'

        # scale down parameters around its best result
        np.random.seed(0)  # Make sure result is consistent
        loc = 1.0
        scale = 0.1
        C = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=3, scale=scale)  # To skip negative values
        loc = 0.1
        scale = 0.2
        gamma = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=3,scale=scale)  # To skip negative values
        kernel = ('rbf', 'linear', 'sigmoid')

        # get the best validated model
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            C=C,
            kernel=kernel,
            gamma=gamma,
            grid_search=True)

        # print all possible parameter values and the best parameters
        svr.print_parameter_candidates()
        svr.print_best_estimator()

        # return the mean squared error
        return svr.mean_sqaured_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_regression(self):
        """
        for dtr, i train on the training data using different :
            1) max_depth
            2) min_samples_leaf
        :return: test accuracy of the dtr best model
        """
        # define parameters
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
#         min_samples_leaf = np.logspace(start=0, stop=4, base=2, num=5, dtype=np.int)
        # best result over max_depth: 4
        # best result over min_samples_leaf: 4

        # scale down parameters around its best result
        max_depth = np.arange(start=3, stop=8, step=2, dtype=np.int)
        min_samples_leaf = np.arange(start=3, stop=8, step=2, dtype=np.int)

        # get the best validated model
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            grid_search=True)

        # print all possible parameter values and the best parameters
        dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        # return the mean squared error
        return dtr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_regression(self):
        """
        for rfr, i train on the training data using different :
            1) n_estimators
            2) max_depth
        :return: test accuracy of the rfr best model
        """
        # define parameters
#         n_estimators = np.logspace(start=8, stop=12, base=2, num=5, dtype=np.int)
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over n_estimators: 1024
        # best result over max_depth: 32

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = norm.rvs(loc=1024, scale=100, size=3).astype(np.int)
        max_depth = norm.rvs(loc=32, scale=10, size=3).astype(np.int)

        # get the best validated model
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_estimators=n_estimators,
            max_depth=max_depth,
            grid_search=True)

        # print all possible parameter values and the best parameters
        rfr.print_parameter_candidates()
        rfr.print_best_estimator()

        # return the mean squared error
        return rfr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def ada_boost_regression(self):
        """
        for abr, i train on the training data using different :
            1) n_estimators
            2) learning_rate
        :return: test accuracy of the abr best model
        """
        # define parameters
#         n_estimators = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
#         learning_rate = np.logspace(start=-2, stop=2, base=2, num=5, dtype=np.float32)
        # best result over n_estimators: 32
        # best result over learning_rate: 1.0

        # scale down parameters around its best result
        # np.random.seed(0)
        n_estimators = norm.rvs(loc=32, scale=6, size=3).astype(np.int)
        learning_rate = norm.rvs(loc=1.0, scale=0.3, size=3).astype(np.float32)

        # Due to the raw model performs better, so we decide to use the raw model

        # get the best validated model
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            # cv=5,
            # n_estimators=n_estimators,
            # learning_rate=learning_rate,
            # grid_search=True
        )

        # print all possible parameter values and the best parameters
        abr.print_parameter_candidates()
        abr.print_best_estimator()

        # return the mean squared error
        return abr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_process_regression(self):
        """
        for gpr, i train on the training data using different :
            1) alpha
            2) kernel
        :return: test accuracy of the gpr best model
        """
        # define parameters
#         kernel = (1.0*RBF(1.0), 1.0*RBF(0.5), WhiteKernel())
#         alpha = np.logspace(start=-2, stop=2, base=2, num=5, dtype=np.float32)
        # best result over kernel: 1**2 * RBF(length_scale=0.5)
        # best result over alpha: 1.0

        # scale down parameters around its best result
        kernel = (1.0 * RBF(1.0), 1.0 * RBF(0.5), WhiteKernel())
        alpha = norm.rvs(loc=1.0, scale=0.5, size=3).astype(np.float32)

        # get the best validated model
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            alpha=alpha,
            kernel=kernel,
            grid_search=True)

        # print all possible parameter values and the best parameters
        gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return gpr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def linear_least_squares(self):
        """
        for lls, i train on the training data using different:
            1) alpha
            2) max_iter
            3) solver
        :return: test accuracy of the lr best model
        """
        # define parameters
#         alpha = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)
#         max_iter = np.logspace(start=2, stop=4, base=10, num=3, dtype=np.int)
#         solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')
        # best result over alpha: 10
        # best result over max_iter: 100
        # best result over solver: 'saga'

        # scale down parameters around its best result
        alpha = norm.rvs(loc=10, scale=2, size=3).astype(np.float32)
        max_iter = norm.rvs(loc=100, scale=20, size=3).astype(np.int)

        # best result over alpha: 10
        # best result over max_iter: 100
        # best result over solver: 'saga'

        # scale down parameters around its best result
        np.random.seed(0)
        alpha = norm.rvs(loc=10, scale=2, size=3).astype(np.float32)
        max_iter = norm.rvs(loc=100, scale=20, size=3).astype(np.int)
        solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')

        # get the best validated model
        lls = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            alpha=alpha,
            max_iter=max_iter,
            solver=solver,
            grid_search=True
        )

        # print all possible parameter values and the best parameters
        lls.print_parameter_candidates()
        lls.print_best_estimator()

        # return the mean squared error
        return lls.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def neural_network_regression(self):
        """
        for nnr, i train on the training data using different :
            1) hidden_layer_sizes
            2) max_iter
        :return: test accuracy of the nnr best model
        """
        # define parameters
#         np.random.seed(0)
#         reciprocal_distribution_hls = scipy.stats.reciprocal(a=100, b=1000)
#         reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
#         hidden_layer_sizes = reciprocal_distribution_hls.rvs(size=10).astype(np.int)
#         max_iter = reciprocal_distribution_mi.rvs(size=10).astype(np.int)
        # best result over hidden_layer_sizes: 779
        # best result over max_iter: 1222

        # scale down parameters around its best result
        np.random.seed(0)
        hidden_layer_sizes = norm.rvs(loc=779, scale=20, size=2).astype(np.int)
        max_iter = norm.rvs(loc=1222, scale=50, size=2).astype(np.int)

        # get the best validated model
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_search=True)

        # print all possible parameter values and the best parameters
        nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # return the mean squared error
        return nnr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)


if __name__ == '__main__':
    wq = Wine_quality()
    print("mean squared error on the actual test set:")
    print('SVR: %.5f' % (wq.support_vector_regression()))
    print('DTR: %.5f' % (wq.decision_tree_regression()))
    print('RFR: %.5f' % (wq.random_forest_regression()))
    print('ABR: %.5f' % (wq.ada_boost_regression()))
    print('GPR: %.5f' % (wq.gaussian_process_regression()))
    print('LLS: %.5f' % (wq.linear_least_squares()))
    print('NNR: %.5f' % (wq.neural_network_regression()))
