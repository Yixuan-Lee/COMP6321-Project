import os
import numpy as np
import pandas as pd
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_regression import Linear_regression
from neural_network_regressor import Neural_network_regressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
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

        # normalize the training set
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
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
        # define arguments given to GridSearchCV
        C = np.logspace(start=-1, stop=3, base=10, num=5,
            dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3,
            dtype=np.float32)  # [0.01, 0.1, 1, 10]
        kernel = ['rbf', 'linear', 'sigmoid']

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
        # define arguments given to GridSearchCV
        max_depth = range(1, 20, 2)
        min_samples_leaf = (1, 20, 2)

        # get the best validated model
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
        n_estimators = range(1, 200, 50)
        max_depth = range(1, 20, 2)

        # get the best validated model
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
        # define arguments given to GridSearchCV
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)   # [0.01, 0.1, 1]

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
            2) kernel (will raise an exception)

        :return: test accuracy of the gpr best model
        :return:
        """
        # define arguments given to GridSearchCV
        # kernel = [DotProduct(), WhiteKernel(), DotProduct() + WhiteKernel()]
        alpha = np.logspace(start=-10, stop=-7, base=10, num=4,
            dtype=np.float32)

        # get the best validated model
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            # kernel=kernel,
            alpha=alpha,
            grid_search=True)

        # print all possible parameter values and the best parameters
        gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return gpr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def linear_regression(self):
        """
        for lr, i train on the training data using same parameters

        :return: test accuracy of the lr best model
        """
        # get the best validated model
        lr = Linear_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3)

        # estimate on the test set
        return lr.mean_squared_error(
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
        # define arguments given to RandomSearchCV
        reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        hidden_layer_sizes = \
            reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
        activation = ['logistic', 'tanh', 'relu']
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)

        # get the best validated model
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_search=True)

        # print all possible parameter values and the best parameters
        nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # # return the mean squared error
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
    print(' LR: %.5f' % cac.linear_regression())
    print('NNR: %.5f' % cac.neural_network_regression())
