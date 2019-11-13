import os
import numpy as np
import pandas as pd
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from scipy.stats import norm    # For tuning parameters
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_least_squares import Linear_least_squares
from neural_network_regressor import Neural_network_regressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import preprocessing
from sklearn.impute import SimpleImputer    # For handling missing values
from sklearn.model_selection import train_test_split


class Communities_and_crime:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/2_Communities_and_Crime'
        filename = 'communities.data'

        # read the data file
        # the first 5 columns are not predictive, so ignore
        total_cols = 128
        used_cols = range(5, total_cols)
        f = pd.read_csv(os.path.join(settings.ROOT_DIR, filepath, filename),
            delimiter=',', usecols=used_cols, header=None, na_values='?',
            dtype=np.float32)   # (1994, 128)
        self.data = f.loc[:, f.columns != total_cols - 1]   # (1994, 122)
        self.targets = f.loc[:, total_cols - 1]  # (1994,)

        # subsampling data (for speeding up)
        # self.data = self.data[:300]
        # self.targets = self.targets[:300]

        # pre-processing strategy 1: ignore the missing-value rows
#         self.data = self.missing_rows_with_missing_values_ignore(self.data)
        # pre-processing strategy 2: impute the missing values with the mean
        self.data = self.missing_rows_with_the_mean(self.data)
        # pre-processing strategy 3: impute the missing values with the most
        # frequent value
        # self.data = self.missing_rows_with_most_frequent_value(self.data)

        # split into the train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set and testing set
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    ##################### Data pre-processing #####################
    def missing_rows_with_missing_values_ignore(self, data):
        # drop the rows with NANs
        return data.dropna()

    def missing_rows_with_the_mean(self, data):
        # train an imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(data)
        # impute the missing values with the mean
        return imp.transform(data)

    def missing_rows_with_most_frequent_value(self, data):
        # train an imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(data)
        # impute the missing values with the most frequent value
        return imp.transform(data)

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
#         C = np.logspace(start=0, stop=3, base=10, num=4, dtype=np.float32)
#         gamma = np.logspace(start=-4, stop=1, base=10, num=4, dtype=np.float32)
#         kernel = ('rbf', 'linear')
        # best result over C: 1.0
        # best result over gamma: 0.004641589
        # best result over kernel: 'rbf'

        # scale down parameters around its best result
        np.random.seed(0)
        C = norm.rvs(loc=1.0, scale=0.1, size=2).astype(np.float32)
        gamma = norm.rvs(loc=0.004641589, scale=0.0004, size=2).astype(np.float32)
        kernel = ('rbf',)

        # get the best validated model
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
#         min_samples_leaf = np.logspace(start=4, stop=7, base=2, num=4, dtype=np.int)
        # best result over max_depth: 8
        # best result over min_samples_leaf: 32

        # scale down parameters around its best result
        np.random.seed(0)
        max_depth = np.arange(start=5, stop=12, step=2, dtype=np.int)
        min_samples_leaf = norm.rvs(loc=32, scale=5, size=3).astype(np.int)

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
        # define arguments given to GridSearchCV
#         n_estimators = np.logspace(start=6, stop=8, base=2, num=3, dtype=np.int)
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over n_estimators: 256
        # best result over max_depth: 32

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = norm.rvs(loc=256, scale=10, size=2).astype(np.int)
        max_depth = norm.rvs(loc=32, scale=4, size=2).astype(np.int)

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
#         n_estimators = np.logspace(start=3, stop=6, base=2, num=4, dtype=np.int)
#         learning_rate = np.logspace(start=-5, stop=-3, base=2, num=3, dtype=np.float32)
        # best result over n_estimators: 64
        # best result over learning_rate:  0.0625

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = norm.rvs(loc=64, scale=5, size=3).astype(np.int)
        learning_rate = norm.rvs(loc=0.0625, scale=0.01, size=3).astype(np.float32)

        # get the best validated model
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

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
#         kernel = (1.0 * RBF(1.0), 1.0 * RBF(0.5), WhiteKernel())
#         alpha = np.logspace(start=-4, stop=-2, base=2, num=3, dtype=np.float32)
        # best result over kernel: 1**2 * RBF(length_scale=1)
        # best result over alpha: 0.0625

        # scale down parameters around its best result
        np.random.seed(0)
        kernel = (1.0 * RBF(1.0), WhiteKernel())
        alpha = norm.rvs(loc=0.0625, scale=0.01, size=2).astype(np.float32)

        # get the best validated model
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            kernel=kernel,
            alpha=alpha,
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
#         max_iter = np.logspace(start=3, stop=4, base=10, num=2, dtype=np.int)
#         solver = ('auto', 'svd', 'saga')
        # best result over alpha: 100.0
        # best result over max_iter: 1000
        # best result over solver: 'svd'

        # scale down parameters around its best result
        np.random.seed(0)
        alpha = norm.rvs(loc=100.0, scale=5, size=2).astype(np.float32)
        max_iter = norm.rvs(loc=1000, scale=100, size=3).astype(np.int)
        solver = ('auto', 'svd', 'saga')

        # get the best validated model
        lls = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
            2) activation
            3) max_iter
        :return: test accuracy of the nnr best model
        """
        # define parameters
#         np.random.seed(0)
#         reciprocal_distribution_hls = scipy.stats.reciprocal(a=100, b=1000)
#         reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
#         hidden_layer_sizes = reciprocal_distribution_hls.rvs(size=5).astype(np.int)
#         max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)
        # best result over hidden_layer_sizes: 519
        # best result over max_iter: 4424

        # scale down parameters around its best result
        np.random.seed(0)
        hidden_layer_sizes = norm.rvs(loc=519, scale=20, size=3).astype(np.int)
        max_iter = norm.rvs(loc=4424, scale=100, size=3).astype(np.int)

        # get the best validated model
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            grid_search=True)

        # print all possible parameter values and the best parameters
        nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # return the mean squared error
        return nnr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)


if __name__ == '__main__':
    cac = Communities_and_crime()
    print("mean squared error on the actual test set:")
    print('SVR: %.5f' % cac.support_vector_regression())
    print('DTR: %.5f' % cac.decision_tree_regression())
    print('RFR: %.5f' % cac.random_forest_regression())
    print('ABR: %.5f' % cac.ada_boost_regression())
    print('GPR: %.5f' % cac.gaussian_process_regression())
    print('LLS: %.5f' % cac.linear_least_squares())
    print('NNR: %.5f' % cac.neural_network_regression())
