import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from scipy.stats import norm    # For tuning parameters
from k_nearest_neighbours import K_nearest_neighbours
from support_vector_classifier import Support_vector_classifier
from decision_tree_classifier import Decision_tree_classifier
from random_forest_classifier import Random_forest_classifier
from ada_boost_classifier import Ada_boost_classifier
from logistic_regression import Logistic_regression
from gaussian_naive_bayes import Gaussian_naive_bayes
from neural_network_classifier import Neural_network_classifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Default_of_credit_card_clients:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/' \
                   '2_Default_of_credit_card_clients/'
        filename = 'default_of_credit_card_clients.xls'

        # read the data file
        # required package: install xlrd
        df = pd.read_excel(io=os.path.join(settings.ROOT_DIR, filepath,
            filename))
        self.data = df.loc[1:, df.columns != 'Y']  # (30000, 24)
        self.targets = df.loc[1:, 'Y'].astype(np.int)  # (30000, )
        # transform DataFrames to numpy.array
        self.data = self.data.to_numpy()
        self.targets = self.targets.to_numpy()

        # randomly sub-sample the dataset
        np.random.seed(0)
        idx = np.arange(self.targets.shape[0])
        np.random.shuffle(idx)
        idx = idx[:1000]
        self.targets = self.targets[idx]
        self.data = self.data[idx]

        # split into the train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set and the testing set
        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    ##################### Model training #####################
    def k_nearest_neighbours(self):
        """
        for knn, i train on the training data using different :
            1) n_neighbors,
            2) weights
        :return: test accuracy of the knn best model
        """
        # define parameters
#         n_neighbors = np.logspace(start=2, stop=6, base=2, num=5, dtype=np.int)
#         weights = ('distance', 'uniform')
        # best result over all n_neighbors: 32
        # best result over all weights: 'distance'

        # scale down parameters around its best result
        np.random.seed(0)
        n_neighbors = norm.rvs(loc=32, scale=10, size=3).astype(np.int)
        weights = ('distance', 'uniform')

        # get the best validated model
        knn = K_nearest_neighbours(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_neighbors=n_neighbors,
            weights=weights,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # knn.print_parameter_candidates()
        # knn.print_best_estimator()

        # return the accuracy score
        return (knn.evaluate(data=self.x_train, targets=self.y_train),
                knn.evaluate(data=self.x_test, targets=self.y_test))

    def support_vector_classifier(self):
        """
        for svc, i train on the training data using different :
            1) C
            2) gamma
            3) kernel
        :return: test accuracy of the svc best model
        """
        # define parameters
#         C = np.logspace(start=0, stop=3, base=10, num=4, dtype=np.int)
#         gamma = np.logspace(start=-4, stop=-1, base=10, num=2, dtype=np.float32)
#         kernel = ('linear', 'rbf')
        # best result over all C:      ?
        # best result over all gamma:  ?
        # best result over all kernel: ?
        # SVC cannot finishes this cross validation in 1 hour!
        # So we decide to use the raw model

        # get the best validated model
        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            # cv=5,
            # n_jobs=3,
            # C=C,
            # gamma=gamma,
            # kernel=kernel,
            # grid_search=True
        )

        # print all possible parameter values and the best parameters
        # svc.print_parameter_candidates()
        # svc.print_best_estimator()

        # return the accuracy score
        return (svc.evaluate(data=self.x_train, targets=self.y_train),
                svc.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_classifier(self):
        """
        for dtc, i train on the training data using different :
            1) criterion
            2) max_depth
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         criterion = ('gini', 'entropy')
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'gini'
        # best result over all max_depth: 4

        # scale down parameters around its best result
        criterion = ('gini', 'entropy')
        scale = 1
        max_depth = np.arange(start=4-scale, stop=4+scale, step=1, dtype=np.int)
        # best result over all criterion: 'gini'
        # best result over all max_depth: 4

        # get the best validated model
        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # dtc.print_parameter_candidates()
        # dtc.print_best_estimator()

        # return the accuracy score
        return (dtc.evaluate(data=self.x_train, targets=self.y_train),
                dtc.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_classifier(self):
        """
        for rfc, i train on the training data using different :
            1) criterion
            2) max_depth
            3) max_depth
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         criterion = ('gini', 'entropy')
#         n_estimators = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'entropy'
        # best result over all n_estimators: 32
        # best result over all max_depth: 8

        # scale down parameters around its best result
        criterion = ('gini', 'entropy')
        scale = 5  # scale of n_estimators
        n_estimators = np.arange(start=32-scale, stop=32+scale, step=3, dtype=np.int)
        scale = 3  # scale of max_depth
        max_depth = np.arange(start=8-scale, stop=8+scale, step=2, dtype=np.int)

        # get the best validated model
        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # rfc.print_parameter_candidates()
        # rfc.print_best_estimator()

        # return the accuracy score
        return (rfc.evaluate(data=self.x_train, targets=self.y_train),
                rfc.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_classifier(self):
        """
        for abc, i train on the training data using different :
            1) n_estimators
            2) learning_rate
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         n_estimators = np.logspace(start=4, stop=8, base=2, num=5, dtype=np.int)
#         learning_rate = np.logspace(start=-4, stop=0, base=10, num=5, dtype=np.float32)
        # best result over all n_estimators: 16
        # best result over all learning_rate: 1e-4

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = norm.rvs(loc=16, scale=5, size=2).astype(np.int)
        learning_rate = norm.rvs(loc=1e-4, scale=1e-5, size=2).astype(np.float32)

        # get the best validated model
        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # abc.print_parameter_candidates()
        # abc.print_best_estimator()

        # return the accuracy score
        return (abc.evaluate(data=self.x_train, targets=self.y_train),
                abc.evaluate(data=self.x_test, targets=self.y_test))

    def logistic_regression(self):
        """
        for lr, i train on the training data using different :
            1) C
            2) max_iter
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         C = np.logspace(start=-3, stop=3, base=10, num=7, dtype=np.float32)
#         max_iter = np.logspace(start=10, stop=13, base=2, num=4, dtype=np.int)
        # best result over all C: 1.0
        # best result over all max_iter: 1024

        # scale down parameters around its best result
        np.random.seed(0)
        scale = 3
        loc = 1.0
        C = loc + scipy.stats.truncnorm.rvs(-loc / scale, np.infty, size=3, scale=scale)  # To skip negative values
        max_iter = norm.rvs(loc=1024, scale=100, size=3).astype(np.int)

        # get the best validated model
        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            C=C,
            max_iter=max_iter,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # lr.print_parameter_candidates()
        # lr.print_best_estimator()

        # return the accuracy score
        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_naive_bayes(self):
        """
        for gnb, i train on the training data using different :
            1) var_smoothing
        :return: test accuracy of the gnb best model
        """
        # define parameters
#         var_smoothing = np.logspace(start=-9, stop=-3, base=10, num=7, dtype=np.float32)
        # best result over all var_smoothing: 0.001

        # scale down parameters around its best result (1st round)
        np.random.seed(0)
        scale = 0.002
        loc = 0.001
        var_smoothing = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=5, scale=scale)  # To skip negative values

        # get the best validated model
        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            var_smoothing=var_smoothing,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # gnb.print_parameter_candidates()
        # gnb.print_best_estimator()

        # return the accuracy score
        return (gnb.evaluate(data=self.x_train, targets=self.y_train),
                gnb.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_classifier(self):
        """
        for nnc, i train on the training data using different :
            1) hidden_layer_sizes
            2) max_iter

        :return: test accuracy of the nnr best model
        """
        # define parameters
#         np.random.seed(0)
#         reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
#         reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
#         hidden_layer_sizes = reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
#         max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)
        # best result over all hidden_layer_sizes: 265
        # best result over all max_iter: 7794

        # scale down parameters around its best result
#         np.random.seed(0)
#         hidden_layer_sizes = norm.rvs(loc=265, scale=50, size=3).astype(np.int)
#         max_iter = norm.rvs(loc=7794, scale=100, size=2).astype(np.int)
        # This cross-validation finished in 15 minutes

        # Due to the reason that the tuned parameters's accuracy is much lower
        # than the raw parameters, we would choose to use the raw parameters


        # get the best random validated model
        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            # cv=3,
            # hidden_layer_sizes=hidden_layer_sizes,
            # max_iter=max_iter,
            # random_search=True
        )

        # print all possible parameter values and best parameters
        # nnc.print_parameter_candidates()
        # nnc.print_best_estimator()

        # return the accuracy score
        return (nnc.evaluate(data=self.x_train, targets=self.y_train),
                nnc.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    doccc = Default_of_credit_card_clients()

    # retrieve the results
    knn_results = doccc.k_nearest_neighbours()
    svc_results = doccc.support_vector_classifier()
    dtc_results = doccc.decision_tree_classifier()
    rfr_results = doccc.random_forest_classifier()
    abc_results = doccc.ada_boost_classifier()
    lr_results = doccc.logistic_regression()
    gnb_results = doccc.gaussian_naive_bayes()
    nnc_results = doccc.neural_network_classifier()

    print("(accuracy, recall, prediction) on training set:")
    print('KNN: (%.3f, %.3f, %.3f)' % (knn_results[0]))
    print('SVC: (%.3f, %.3f, %.3f)' % (svc_results[0]))
    print('DTC: (%.3f, %.3f, %.3f)' % (dtc_results[0]))
    print('RFC: (%.3f, %.3f, %.3f)' % (rfr_results[0]))
    print('ABC: (%.3f, %.3f, %.3f)' % (abc_results[0]))
    print(' LR: (%.3f, %.3f, %.3f)' % (lr_results[0]))
    print('GNB: (%.3f, %.3f, %.3f)' % (gnb_results[0]))
    print('NNC: (%.3f, %.3f, %.3f)' % (nnc_results[0]))

    print("(accuracy, recall, prediction) on testing set:")
    print('KNN: (%.3f, %.3f, %.3f)' % (knn_results[1]))
    print('SVC: (%.3f, %.3f, %.3f)' % (svc_results[1]))
    print('DTC: (%.3f, %.3f, %.3f)' % (dtc_results[1]))
    print('RFC: (%.3f, %.3f, %.3f)' % (rfr_results[1]))
    print('ABC: (%.3f, %.3f, %.3f)' % (abc_results[1]))
    print(' LR: (%.3f, %.3f, %.3f)' % (lr_results[1]))
    print('GNB: (%.3f, %.3f, %.3f)' % (gnb_results[1]))
    print('NNC: (%.3f, %.3f, %.3f)' % (nnc_results[1]))
