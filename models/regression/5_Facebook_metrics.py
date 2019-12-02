import os
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from models import settings

import warnings

from models.regression.ada_boost_regressor import Ada_boost_regressor
from models.regression.decision_tree_regressor import Decision_tree_regressor
from models.regression.gaussian_process_regressor import Gaussian_process_regressor
from models.regression.linear_least_squares import Linear_least_squares
from models.regression.neural_network_regressor import Neural_network_regressor
from models.regression.random_forest_regressor import Random_forest_regressor
from models.regression.support_vector_regressor import Support_vector_regressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignore sklearn convergence warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore sklearn deprecation warnings


class Facebook_metrics:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/5_Facebook_metrics'
        filename = 'dataset_Facebook.csv'

        # read data from the source file
        f = lambda s: (0 if s == 'Photo' else (1 if s == 'Status' else (2 if s == 'Link' else 3)))
        r = pd.read_csv(os.path.join(settings.ROOT_DIR, filepath,
                                     filename), sep=';', converters={1: f})
        r = self.missing_rows_with_most_frequent_value(r).astype(np.int)
        self.targets = r[:, 7:]
        self.data = r[:, :7]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                             random_state=0)

        # datasets normalization
        # train_matrix = np.column_stack((self.x_train, self.y_train))
        # test_matrix = np.column_stack((self.x_test, self.y_test))
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        # self.y_train = scaler.transform(transformed_train_m[:, 7:])
        # self.x_train = scaler.transform(transformed_train_m[:, :7])
        # self.y_test = scaler.transform(transformed_test_m)

    def printself(self):
        print(self.data)
        print(self.targets)

    ##################### Data pre-processing (by Yixuan Li)#####################
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

    ###################### model training ######################
    def support_vector_regression(self, y_train, y_test):
        # coef0 = scipy.stats.uniform(0, 5)
        # epsilon = scipy.stats.reciprocal(0.01, 1)

        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=30,
            n_jobs=10,
            C=scipy.stats.reciprocal(1, 100),
            kernel=['sigmoid', 'rbf', 'linear'],
            gamma=scipy.stats.reciprocal(0.01, 20),
            random_search=True)

        svr.print_parameter_candidates()
        svr.print_best_estimator()

        return svr.mean_sqaured_error(
            x_test=self.x_test,
            y_test=y_test)

    def decision_tree_regression(self, y_train, y_test):
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=50,
            max_depth=range(1, 20),
            min_samples_leaf=range(1, 20),
            n_jobs=10,
            random_search=True)

        dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return dtr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)

    def random_forest_regression(self, y_train, y_test):
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=50,
            n_jobs=10,
            n_estimators=range(1, 100),
            max_depth=range(1, 20),
            random_search=True)

        rfr.print_parameter_candidates()
        rfr.print_best_estimator()

        return rfr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)

    def ada_boost_regression(self, y_train, y_test):
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=99,
            n_estimators=range(1, 100),
            n_jobs=10,
            random_search=True)

        abr.print_parameter_candidates()
        abr.print_best_estimator()

        return abr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)

    def gaussian_process_regression(self, y_train, y_test):
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=10,
            # kernel=kernel,
            alpha=scipy.stats.reciprocal(1e-11, 1e-8),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return gpr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)

    def linear_regression(self, y_train, y_test):
        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=y_train,
            n_jobs = 10,

        )

        return lr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)

    def neural_network_regression(self, y_train, y_test):
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=y_train,
            cv=3,
            n_iter=1,
            hidden_layer_sizes=range(100, 1000),
            activation=['logistic', 'tanh', 'relu'],
            max_iter=range(1000, 10000),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # # return the mean squared error
        return nnr.mean_squared_error(
            x_test=self.x_test,
            y_test=y_test)


if __name__ == '__main__':
    fm = Facebook_metrics()
    outputfile = open('5_result.txt',"w")
    str_list = []
    str_list.append("mean squared error on the actual test set:"+os.linesep)
    for i in range(12):
        str_list.append('Output feature [%d] of SVR: %.5f' % (
            i, fm.support_vector_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
        str_list.append(os.linesep)
    #     str_list.append('Output feature [%d] of DTR: %.5f' % (
    #         i, fm.decision_tree_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
    #     str_list.append(os.linesep)
    #     str_list.append('Output feature [%d] of RFR: %.5f' % (
    #         i, fm.random_forest_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
    #     str_list.append(os.linesep)
    #     str_list.append('Output feature [%d] of DTR: %.5f' % (
    #         i, fm.ada_boost_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
    #     str_list.append(os.linesep)
    #     str_list.append('Output feature [%d] of GPR: %.5f' % (
    #         i, fm.gaussian_process_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
    #     str_list.append(os.linesep)
    #     str_list.append('Output feature [%d] of  LR: %.5f' % (
    #         i, fm.linear_regression(y_train=fm.y_train, y_test=fm.y_test[:, i])))
    #     str_list.append(os.linesep)
    # str_list.append('NNR: %.5f' % (fm.neural_network_regression(y_train=fm.y_train, y_test=fm.y_test)))
    # str_list.append(os.linesep)
    outputfile.write("".join(str_list))
    outputfile.close()
