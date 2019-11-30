import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from scipy.io import arff       # For loading .arff file
from scipy.stats import norm    # For tuning parameters
from k_nearest_neighbours import K_nearest_neighbours
from support_vector_classifier import Support_vector_classifier
from decision_tree_classifier import Decision_tree_classifier
from random_forest_classifier import Random_forest_classifier
from ada_boost_classifier import Ada_boost_classifier
from logistic_regression import Logistic_regression
from gaussian_naive_bayes import Gaussian_naive_bayes
from neural_network_classifier import Neural_network_classifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Diabetic_retinopathy:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/1_Diabetic_Retinopathy'
        filename = 'messidor_features.arff'

        # read the data file
        file, meta = arff.loadarff(os.path.join(settings.ROOT_DIR, filepath,
            filename))
        file_data = np.asarray(file.tolist(), dtype=np.float32)
        self.data = file_data[:, :-1]   # (1151, 19)
        self.targets = file_data[:, -1]  # (1151, )

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
            1) n_neighbors
            2) weights

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         n_neighbors = np.logspace(start=1, stop=9, base=2, num=9, dtype=np.int)
#         weights = ('distance', 'uniform')
        # best result over all n_neighbors: 64
        # best result over all weights: 'distance'

        # scale down parameters around its best result (1st)
#         n_neighbors = norm.rvs(loc=64, scale=32, size=20).astype(np.int)
#         weights = ('distance', 'uniform')
        # best result over all n_neighbors: 69
        # best result over all weights: 'distance'

        # scale down parameters around its best result (2nd)
        scale = 5
        n_neighbors = np.arange(start=69-scale, stop=69+scale, step=1, dtype=np.int)
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

        return (knn.evaluate(data=self.x_train, targets=self.y_train),
                knn.evaluate(data=self.x_test, targets=self.y_test))

    def support_vector_classifier(self):
        """
        for svc, i train on the training data using different :
            1) C
            2) gamma
            3) kernel

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         C = np.logspace(start=0, stop=3, base=10, num=4, dtype=np.int)
#         gamma = np.logspace(start=-4, stop=-1, base=10, num=4, dtype=np.float32)
#         kernel = ('linear', 'rbf', 'sigmoid', 'poly')
        # best result over all C: 100
        # best result over all gamma: 1e-4
        # best result over all kernel: 'linear'

        # scale down parameters around its best result (1st)
#         scale = 50   # scale of C for scipy.stats.rvs
#         loc = 100    # mean of C for scipy.stats.rvs
#         C = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=10, scale=scale)  # To skip negative values
#         scale = 1e-3  # scale of gamma for scipy.stat.rvs
#         loc = 1e-4    # mean of gamma for scipy.stats.rvs
#         gamma = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=10, scale=scale)  # To skip negative values
#         kernel = ('linear', 'rbf', 'sigmoid', 'poly')
        # best result over all C: 113.4405557380417
        # best result over all gamma: 0.0011705279015427493
        # best result over all kernel: 'linear'

        # scale down parameters around its best result (2nd)
        np.random.seed(0)
        scale = 5
        loc = 113.4405557380417
        C = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=5, scale=scale)  # To skip negative values
        scale = 1e-4
        loc = 0.0011705279015427493
        gamma = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=5, scale=scale)  # To skip negative values
        kernel = ('linear', 'rbf', 'sigmoid', 'poly')

        # get the best validated model
        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            C=C,
            gamma=gamma,
            kernel=kernel,
            random_search=True)

        # print all possible parameter values and the best parameters
        # svc.print_parameter_candidates()
        # svc.print_best_estimator()

        return (svc.evaluate(data=self.x_train, targets=self.y_train),
                svc.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_classifier(self):
        """
        for dtc, i train on the training data using different :
            1) criterion
            2) max_depth

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         criterion = ('gini', 'entropy')
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'entropy'
        # best result over all max_depth: 8

        # scale down parameters around its best result
#         criterion = ('gini', 'entropy')
#         scale = 4
#         max_depth = np.arange(start=8-scale, stop=8+scale, step=1, dtype=np.int)
        # best result over all criterion: 'entropy'
        # best result over all max_depth: 7

        # Due to the reason that the tuned parameters's accuracy is much lower
        # than the raw parameters, we would choose to use the raw parameters

        # get the best validated model
        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
#             cv=5,
#             criterion=criterion,
#             max_depth=max_depth,
            grid_search=True)

        # print all possible parameter values and the best parameters
        # dtc.print_parameter_candidates()
        # dtc.print_best_estimator()

        return (dtc.evaluate(data=self.x_train, targets=self.y_train),
                dtc.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_classifier(self):
        """
        for rfc, i train on the training data using different :
            1) criterion
            2) n_estimators
            3) max_depth

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         criterion = ('gini', 'entropy')
#         n_estimators = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'gini'
        # best result over all n_estimators: 32
        # best result over all max_depth: 32

        # scale down parameters around its best result
        criterion = ('gini', 'entropy')
        scale = 2  # scale of n_estimators
        n_estimators = np.arange(start=32-scale, stop=32+scale, step=1, dtype=np.int)
        scale = 3  # scale of max_depth
        max_depth = np.arange(start=32-scale, stop=32+scale, step=1, dtype=np.int)

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

        return (rfc.evaluate(data=self.x_train, targets=self.y_train),
                rfc.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_classifier(self):
        """
        for abc, i train on the training data using different :
            1) n_estimators
            2) learning_rate

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         n_estimators = np.logspace(start=1, stop=8, base=2, num=8, dtype=np.int)
#         learning_rate = np.logspace(start=-4, stop=0, base=10, num=5, dtype=np.float32)
        # best result over all n_estimators: 64
        # best result over all learning_rate: 1.0

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = norm.rvs(loc=64, scale=16, size=5).astype(np.int)
        learning_rate = norm.rvs(loc=1.0, scale=0.5, size=5).astype(np.float32)

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

        return (abc.evaluate(data=self.x_train, targets=self.y_train),
                abc.evaluate(data=self.x_test, targets=self.y_test))

    def logistic_regression(self):
        """
        for lr, i train on the training data using different :
            1) C
            2) max_iter

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         C = np.logspace(start=-3, stop=3, base=10, num=7, dtype=np.float32)
#         max_iter = np.logspace(start=10, stop=15, base=2, num=8, dtype=np.int)
        # best result over all C: 100.0
        # best result over all max_iter: 1024

        # scale down parameters around its best result
        np.random.seed(0)
        C = norm.rvs(loc=100.0, scale=50.0, size=5).astype(np.float32)
        max_iter = norm.rvs(loc=1024, scale=100, size=5).astype(np.int)

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

        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_naive_bayes(self):
        """
        for gnb, i train on the training data using different :
            1) var_smoothing

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         var_smoothing = np.logspace(start=-9, stop=-3, base=10, num=7, dtype=np.float32)
        # best result over all var_smoothing: 1e-08

        # scale down parameters around its best result
        np.random.seed(0)
        var_smoothing = norm.rvs(loc=1e-8, scale=1e-8, size=5).astype(np.float32)

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

        :return: ((accuracy_train, recall_train, precision_train),
                  (accuracy_test,  recall_test,  precision_test))
        """
        # define parameters
#         np.random.seed(0)
#         reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
#         reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
#         hidden_layer_sizes = reciprocal_distrobution_hls.rvs(size=10).astype(np.int)
#         max_iter = reciprocal_distribution_mi.rvs(size=10).astype(np.int)
        # best result over all hidden_layer_sizes: 919
        # best result over all max_iter: 1047

        # scale down parameters around its best result
        np.random.seed(0)
        hidden_layer_sizes = norm.rvs(loc=919, scale=10, size=3).astype(np.int)
        max_iter = norm.rvs(loc=1047, scale=50, size=3).astype(np.int)

        # get the best random validated model
        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_search=True)

        # print all possible parameter values and best parameters
        # nnc.print_parameter_candidates()
        # nnc.print_best_estimator()

        # return the accuracy score
        return (nnc.evaluate(data=self.x_train, targets=self.y_train),
                nnc.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    dr = Diabetic_retinopathy()

    # retrieve the results
    knn_results = dr.k_nearest_neighbours()
    svc_results = dr.support_vector_classifier()
    dtc_results = dr.decision_tree_classifier()
    rfr_results = dr.random_forest_classifier()
    abc_results = dr.ada_boost_classifier()
    lr_results = dr.logistic_regression()
    gnb_results = dr.gaussian_naive_bayes()
    nnc_results = dr.neural_network_classifier()

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
