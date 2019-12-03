import os
import numpy as np
import xlrd
import scipy.stats
from pathlib import Path
import pandas as pd
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_least_squares import Linear_least_squares
from neural_network_regressor import Neural_network_regressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import preprocessing


class Student_Performance:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    def __init__(self):
        filepath = 'datasets/regression_datasets/7_Student_Performance/'
        # filename1 = 'student-mat.csv'
        filename2 = 'student-por.csv'

        # read the 2 data files
        # file1 = np.loadtxt(os.path.join(filepath, filename1),
            # delimiter=';', dtype=np.object, skiprows=1)
        file2 = np.loadtxt(os.path.join(filepath, filename2),
            delimiter=';', dtype=np.object, skiprows=1)
        self.data = np.asarray(file2.tolist())
        self.data = file2[:, :-1]
        self.targets = file2[:, -1]
        numeric = self.data[:, np.r_[2, 6, 7, 12:14, 23:30]]
        self.data = np.delete(self.data,[2, 6, 7, 12, 13, 14,
            23, 24 ,25 ,26 ,27, 28, 29, 30],axis=1)
        encode = preprocessing.OneHotEncoder().fit(self.data)
        self.data = encode.transform(self.data).toarray()
        self.data = np.column_stack((self.data, numeric))
        self.data = np.asarray(self.data, dtype=np.float32)

        # 2 6 7  12 13 14   23 ~ 32 
        # 0 1 3 4 5 8 9 10 11 15 16 17 18 19 20 21 22

        # split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

    def support_vector_regression(self):
        C = np.logspace(start=-1, stop=3, base=10, num=5,
            dtype=np.float32)
        gamma = np.logspace(start=-1, stop=1, base=10, num=3,
            dtype=np.float32)
        kernel = ['rbf', 'linear', 'sigmoid']

        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
            C=C,
            kernel=kernel,
            gamma=gamma,
            grid_search=True)

        # svr.print_parameter_candidates()
        # svr.print_best_estimator()

        return (svr.evaluate(data=self.x_train, targets=self.y_train),
                svr.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_regression(self):
        max_depth = range(1, 20, 2)
        min_samples_leaf = (1, 20, 2)

        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            grid_search=True)

        # dtr.print_parameter_candidates()
        # dtr.print_best_estimator()

        return (dtr.evaluate(data=self.x_train, targets=self.y_train),
                dtr.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_regression(self):
        n_estimators = range(1, 200, 50)
        max_depth = range(1, 20, 2)

        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
            n_estimators=n_estimators,
            max_depth=max_depth,
            grid_search=True)

        # rfr.print_parameter_candidates()
        # rfr.print_best_estimator()

        return (rfr.evaluate(data=self.x_train, targets=self.y_train),
                rfr.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_regression(self):
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)   # [0.01, 0.1, 1]

        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_jobs=10,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

        # abr.print_parameter_candidates()
        # abr.print_best_estimator()

        return (abr.evaluate(data=self.x_train, targets=self.y_train),
                abr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_process_regression(self):
        alpha = np.logspace(start=-10, stop=-7, base=10, num=4,
            dtype=np.float32)

        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            alpha=alpha,
            grid_search=True)

        # gpr.print_parameter_candidates()
        # gpr.print_best_estimator()

        return (gpr.evaluate(data=self.x_train, targets=self.y_train),
                gpr.evaluate(data=self.x_test, targets=self.y_test))

    def linear_least_squares(self):
        np.random.seed(0)
        alpha = norm.rvs(loc=64, scale=2, size=3).astype(np.float32)
        max_iter = norm.rvs(loc=100, scale=20, size=3).astype(np.int)
        solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')

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
        # lls.print_parameter_candidates()
        # lls.print_best_estimator()

        return (lls.evaluate(data=self.x_train, targets=self.y_train),
                lls.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_regression(self):
        reciprocal_distribution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        hidden_layer_sizes = \
            reciprocal_distribution_hls.rvs(size=5).astype(np.int)
        activation = ['logistic', 'tanh', 'relu']
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)

        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_search=True)

        # nnr.print_parameter_candidates()
        # nnr.print_best_estimator()

        return (nnr.evaluate(data=self.x_train, targets=self.y_train),
                nnr.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    sp = Student_Performance()

    svr_results = sp.support_vector_regression()
    dtr_results = sp.decision_tree_regression()
    rfr_results = sp.random_forest_regression()
    abr_results = sp.ada_boost_regression()
    gpr_results = sp.gaussian_process_regression()
    lls_results = sp.linear_least_squares()
    nnr_results = sp.neural_network_regression()

    print("(mean_square_error, r2_score) on training set:")
    print('SVR: (%.3f, %.3f)' % (svr_results[0]))
    print('DTR: (%.3f, %.3f)' % (dtr_results[0]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[0]))
    print('ABR: (%.3f, %.3f)' % (abr_results[0]))
    print('GPR: (%.3f, %.3f)' % (gpr_results[0]))
    print('LLS: (%.3f, %.3f)' % (lls_results[0]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[0]))

    print("(mean_square_error, r2_score) on test set:")
    print('SVR: (%.3f, %.3f)' % (svr_results[1]))
    print('DTR: (%.3f, %.3f)' % (dtr_results[1]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[1]))
    print('ABR: (%.3f, %.3f)' % (abr_results[1]))
    print('GPR: (%.3f, %.3f)' % (gpr_results[1]))
    print('LLS: (%.3f, %.3f)' % (lls_results[1]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[1]))