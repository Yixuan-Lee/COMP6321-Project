import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
import sklearn.model_selection  # for RandomizedSearchCV
from models import settings     # for retrieving root path
from sklearn.svm import SVR
from support_vector_regressor import Support_vector_regressor
from sklearn.tree import DecisionTreeRegressor
from decision_tree_regressor import Decision_tree_regressor
from sklearn.ensemble import RandomForestRegressor
from random_forest_regressor import Random_forest_regressor
from sklearn.ensemble import AdaBoostRegressor
from ada_boost_regressor import Ada_boost_regressor

from gaussian_process_regressor import Gaussian_process_regressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from linear_regression import Linear_regression
from sklearn.neural_network import MLPRegressor
from neural_network_regressor import Neural_network_regressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

        # split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = self.scaler.transform(self.x_test)

    ##################### Model training #####################
    def support_vector_regression(self):
        """
        for svr, i train on the training data using different :
            1) C
            2) gamma

        :return: test accuracy of the svr best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        C = np.logspace(start=-1, stop=3, base=10, num=5,
            dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3,
            dtype=np.float32)  # [0.01, 0.1, 1, 10]
        param_grid = {
            'C': C,
            'gamma': gamma,
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print(param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return squared_error

    def decision_tree_regression(self):
        """
        for dtr, i train on the training data using different :
            1) criterion
            2) max_depth

        :return: test accuracy of the dtr best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        criterion = ['mse', 'mae']
        max_depth = range(1, 20, 2)
        param_grid = {
            'criterion': criterion,
            'max_depth': max_depth
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print(param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return squared_error

    def random_forest_regression(self):
        """
        for rfr, i train on the training data using different :
            1) n_estimators
            2) max_depth

        :return: test accuracy of the rfr best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        n_estimators = range(1, 200, 50)
        max_depth = range(1, 20, 2)
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=RandomForestRegressor(random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print(param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return squared_error



    def ada_boost_regression(self):
        """
        for abr, i train on the training data using different :
            1) n_estimators
            2) learning_rate

        :return: test accuracy of the abr best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)   # [0.01, 0.1, 1]
        param_grid = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=AdaBoostRegressor(random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print(param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return squared_error


    def gaussian_process_regression(self):
        """
        for gpr, i train on the training data using different :
            1)
            2)

        :return: test accuracy of the gpr best model
        :return:
        """
        kernel = DotProduct() + WhiteKernel()
        gpr = Gaussian_process_regressor(k=kernel)
        gpr.train(self.x_train, self.y_train)
        return gpr.get_mse(self.x_test, self.y_test)

    def linear_regression(self):
        """
        for lr, i train on the training data using same parameters

        :return: test accuracy of the lr best model
        """

        lr = LinearRegression()
        lr.fit(self.x_train, self.y_train)

        # estimate on the test set
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=lr.predict(self.x_test))

        return squared_error

    def neural_network_regression(self):
        """
        for nnr, i train on the training data using different :
            1) hidden_layer_sizes
            2) max_iter

        :return: test accuracy of the nnr best model
        """

        # RandomSearch Cross Validation
        # define param_dist argument to give RandomSearchCV
        reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        reci_hidden_layer_sizes = \
            reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)
        param_dist = {
            'hidden_layer_sizes': reci_hidden_layer_sizes,
            'max_iter': max_iter
        }

        # fit a 5-fold RandomSearchCV instance
        rscv = sklearn.model_selection.RandomizedSearchCV(
            estimator=MLPRegressor(activation='relu', random_state=0),
            param_distributions=param_dist,
            verbose=1,
            cv=5,
            n_jobs=5,   # 5 concurrent workers
            n_iter=5,
            random_state=0)
        rscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print(param_dist)
        print(rscv.best_estimator_)

        # estimate on the test set
        best_rscv = rscv.best_estimator_
        squared_error = mean_squared_error(
            y_true=self.y_test,
            y_pred=best_rscv.predict(self.x_test))

        return squared_error


if __name__ == '__main__':
    wq = Wine_quality()
    print("mean squared error on the actual test set:")
    # print('SVR: %.5f' % (wq.support_vector_regression()))
    # print('DTR: %.5f' % wq.decision_tree_regression())
    # print('RFR: %.5f' % (wq.random_forest_regression()))
    # print('ABR: %.5f' % (wq.ada_boost_regression()))

    # print('GPR: %.5f' % wq.gaussian_process_regression())

    # print(' LR: %.5f' % wq.linear_regression())
    print('NNR: %.5f' % wq.neural_network_regression())
