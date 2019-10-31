import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from linear_regression import Linear_regression
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
        self.data = self.data[:1000]
        self.targets = self.targets[:1000]


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
            3) kernel
        :return: test accuracy of the svr best model
        """
        # define arguments given to GridSearchCV
        C = np.logspace(start=-1, stop=3, base=10, num=5,
            dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3,
            dtype=np.float32)  # [0.1, 1, 10]
        kernel = ['rbf', 'linear', 'sigmoid']

        # get the best validated model
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
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
            n_jobs=10,
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
            n_jobs=10,
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
            n_jobs=10,
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
            n_jobs=10,
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
            cv=3,
            n_jobs=10)

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
        reciprocal_distribution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        hidden_layer_sizes = \
            reciprocal_distribution_hls.rvs(size=5).astype(np.int)
        activation = ['logistic', 'tanh', 'relu']
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)

        # get the best validated model
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
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
    wq = Wine_quality()
    print("mean squared error on the actual test set:")
    print('SVR: %.5f' % (wq.support_vector_regression()))
    print('DTR: %.5f' % (wq.decision_tree_regression()))
    print('RFR: %.5f' % (wq.random_forest_regression()))
    print('ABR: %.5f' % (wq.ada_boost_regression()))

    # iterating over kernels will raise a bug
    print('GPR: %.5f' % (wq.gaussian_process_regression()))

    print(' LR: %.5f' % (wq.linear_regression()))
    print('NNR: %.5f' % (wq.neural_network_regression()))
