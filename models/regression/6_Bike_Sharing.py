import os
import numpy as np
import scipy.stats
import sklearn
from sklearn.model_selection import train_test_split

from models import settings
from models.regression.ada_boost_regressor import Ada_boost_regressor
from models.regression.decision_tree_regressor import Decision_tree_regressor
from models.regression.gaussian_process_regressor import Gaussian_process_regressor

from models.regression.linear_least_squares import Linear_least_squares
from models.regression.neural_network_regressor import Neural_network_regressor
from models.regression.random_forest_regressor import Random_forest_regressor
from models.regression.support_vector_regressor import Support_vector_regressor


class Bike_Sharing:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/6_Bike_Sharing'
        filename = 'hour.csv'

        # read data from the source file
        self.data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
                                            filename), delimiter=',', usecols=range(2, 17), skiprows=1)
        self.targets = self.data[:, -1]
        self.data = self.data[:, :-1]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                             random_state=0)

        # normalize the training set
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)

    def support_vector_regression(self):

        # coef0 = scipy.stats.uniform(0, 5)
        # epsilon = scipy.stats.reciprocal(0.01, 1)

        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            n_jobs=10,
            C= scipy.stats.reciprocal(1, 100),
            kernel=['sigmoid', 'rbf' , 'linear'],
            gamma= scipy.stats.reciprocal(0.01, 20),
            random_search=True)

        svr.print_parameter_candidates()
        svr.print_best_estimator()

        return svr.mean_sqaured_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_regression(self):

        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=50,
            max_depth=range(1, 20),
            min_samples_leaf= range(1,20),
            n_jobs=10,
            random_search=True)

        dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return dtr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_regression(self):

        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
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
            y_test=self.y_test)

    def ada_boost_regression(self):

        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=99,
            n_estimators=range(1, 100),
            n_jobs=10,
            random_search=True)

        abr.print_parameter_candidates()
        abr.print_best_estimator()

        return abr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_process_regression(self):
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            # kernel=kernel,
            alpha=scipy.stats.reciprocal(1e-11,1e-8),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return gpr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def linear_regression(self):
        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            # n_jobs = 10
        )

        return lr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)

    def neural_network_regression(self):
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            hidden_layer_sizes=range(100,1000),
            activation=['logistic', 'tanh', 'relu'],
            max_iter=range(1000,10000),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # # return the mean squared error
        return nnr.mean_squared_error(
            x_test=self.x_test,
            y_test=self.y_test)


    def printself(self):
        print(self.data)
        print(self.targets)


if __name__ == '__main__':
    bs = Bike_Sharing()
    # bs.printself()
    print("mean squared error on the actual test set:")
    print('SVR: %.5f' % bs.support_vector_regression())
    print('DTR: %.5f' % bs.decision_tree_regression())
    print('RFR: %.5f' % bs.random_forest_regression())
    print('ABR: %.5f' % bs.ada_boost_regression())
    print('GPR: %.5f' % bs.gaussian_process_regression())
    print(' LR: %.5f' % bs.linear_regression())
    print('NNR: %.5f' % bs.neural_network_regression())
    

