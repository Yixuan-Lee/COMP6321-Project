import os
import numpy as np
import pandas as pd
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


class Seismic_bumps:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/10_Seismic_Bumps'
        filename = 'seismic-bumps.arff'

        # read the data file
        file, meta = arff.loadarff(os.path.join(settings.ROOT_DIR, filepath,
            filename))
        df = pd.DataFrame(file)
        # remove the prefix 'b' in columns
        df['seismic'] = df['seismic'].str.decode('utf-8')
        df['seismoacoustic'] = df['seismoacoustic'].str.decode('utf-8')
        df['shift'] = df['shift'].str.decode('utf-8')
        df['ghazard'] = df['ghazard'].str.decode('utf-8')
        df['class'] = df['class'].str.decode('utf-8')
        # transform characters to numbers
        mapping = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'N': 1,
            'W': 0
        }
        df = df.replace({'seismic': mapping,
                         'seismoacoustic': mapping,
                         'shift': mapping,
                         'ghazard': mapping})
        # transform the column's dtype
        df['seismic'] = df['seismic'].astype(np.int64)
        df['seismoacoustic'] = df['seismoacoustic'].astype(np.int64)
        df['shift'] = df['shift'].astype(np.int64)
        df['ghazard'] = df['ghazard'].astype(np.int64)
        df['class'] = df['class'].astype(np.int64)
        file_data = df.to_numpy()
        self.data = file_data[:, :-1]   # (2584, 18)
        self.targets = file_data[:, -1]   # (2584, )

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
        :return: test accuracy of the knn best model
        """
        # define parameters
#         n_neighbors = np.logspace(start=1, stop=9, base=2, num=9, dtype=np.int)
#         weights = ('distance', 'uniform')
        # best result over all n_neighbors: 32
        # best result over all weights: 'distance'

        # scale down parameters around its best result
        np.random.seed(0)
        n_neighbors = norm.rvs(loc=32, scale=8, size=5).astype(np.int)
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
        knn.print_parameter_candidates()
        knn.print_best_estimator()

        # return the accuracy score
        return knn.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def support_vector_classifier(self):
        """
        for svc, i train on the training data using different :
            1) C
            2) gamma
            3) kernel
        :return: test accuracy of the svc best model
        """
        # define parameters
#         C = np.logspace(start=0, stop=2, base=10, num=3, dtype=np.int)
#         gamma = np.logspace(start=-4, stop=-2, base=10, num=3, dtype=np.float32)
#         kernel = ('linear', 'rbf')
        # best result over all C: 1
        # best result over all gamma: 1e-04
        # best result over all kernel: 'linear'

        # scale down parameters around its best result
        np.random.seed(0)
        scale = 0.5
        loc = 1.0
        C = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=2, scale=scale)  # To skip negative values
        scale = 1e-4
        loc = 1e-4
        gamma = loc + scipy.stats.truncnorm.rvs(-loc/scale, np.infty, size=2, scale=scale)  # To skip negative values
        kernel = ('linear',)

        # get the best validated model
        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            gamma=gamma,
            kernel=kernel,
            grid_search=True)

        # print all possible parameter values and the best parameters
        svc.print_parameter_candidates()
        svc.print_best_estimator()

        # return the accuracy score
        return svc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_classifier(self):
        """
        for dtc, i train on the training data using different :
            1) criterion
            2) max_depth
        :return: test accuracy of the dtc best model
        """
        # define parameters
        criterion = ('gini', 'entropy')
        max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'entropy'
        # best result over all max_depth: 2

        # scale down parameters around its best result
        criterion = ('gini', 'entropy')
        scale = 1
        max_depth = np.arange(start=2-scale, stop=2+scale, step=1, dtype=np.int)

        # get the best validated model
        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        # print all possible parameter values and the best parameters
        dtc.print_parameter_candidates()
        dtc.print_best_estimator()

        # return the accuracy score
        return dtc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_classifier(self):
        """
        for rfc, i train on the training data using different :
            1) criterion
            2) n_estimators
            3) max_depth
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         criterion = ('gini', 'entropy')
#         n_estimators = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
#         max_depth = np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int)
        # best result over all criterion: 'entropy'
        # best result over all n_estimators: 16
        # best result over all max_depth: 32

        # scale down parameters around its best result
#         criterion = ('gini', 'entropy')
#         scale = 5  # scale of n_estimators
#         n_estimators = np.arange(start=16-scale, stop=16+scale, step=2, dtype=np.int)
#         scale = 8  # scale of max_depth
#         max_depth = np.arange(start=32-scale, stop=32+scale, step=3, dtype=np.int)

        # (Raw model performs better, we decide to use raw model)

        # get the best validated model
        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            # cv=5,
            # criterion=criterion,
            # n_estimators=n_estimators,
            # max_depth=max_depth,
            # grid_search=True
        )

        # print all possible parameter values and the best parameters
        rfc.print_parameter_candidates()
        rfc.print_best_estimator()

        # return the accuracy score
        return rfc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def ada_boost_classifier(self):
        """
        for abc, i train on the training data using different :
            1) n_estimators
            2) learning_rate
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         n_estimators = np.logspace(start=1, stop=8, base=2, num=8, dtype=np.int)
#         learning_rate = np.logspace(start=-4, stop=0, base=10, num=5, dtype=np.float32)
        # best result over all n_estimators: 2
        # best result over all learning_rate: 1e-04

        # scale down parameters around its best result
        np.random.seed(0)
        n_estimators = np.arange(start=1, stop=3, step=1, dtype=np.int)
        learning_rate = norm.rvs(loc=1e-04, scale=1e-04, size=5).astype(np.float32)

        # get the best validated model
        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

        # print all possible parameter values and the best parameters
        abc.print_parameter_candidates()
        abc.print_best_estimator()

        # return the accuracy score
        return abc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def logistic_regression(self):
        """
        for lr, i train on the training data using different :
            1) C
            2) max_iter
        :return: test accuracy of the dtc best model
        """
        # define parameters
#         C = np.logspace(start=-6, stop=-3, base=10, num=4, dtype=np.float32)
#         max_iter = np.logspace(start=5, stop=8, base=2, num=4, dtype=np.int)
        # best result over all C: 1e-06
        # best result over all max_iter: 32

        # scale down parameters around its best result
        np.random.seed(0)
        C = norm.rvs(loc=1e-06, scale=1e-06, size=5).astype(np.float32)
        max_iter = norm.rvs(loc=32, scale=12, size=5).astype(np.int)

        # get the best validated model
        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            C=C,
            max_iter=max_iter,
            grid_search=True)

        # print all possible parameter values and the best parameters
        lr.print_parameter_candidates()
        lr.print_best_estimator()

        # return the accuracy score
        return lr.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_naive_bayes(self):
        """
        for gnb, i train on the training data using different :
            1) var_smoothing

        :return: test accuracy of the gnb best model
        """
        # define parameters
#         var_smoothing = np.logspace(start=-3, stop=2, base=10, num=6, dtype=np.float32)
        # best result over all var_smoothing: 100.0

        # scale down parameters around its best result
        np.random.seed(0)
        var_smoothing = norm.rvs(loc=100.0, scale=20, size=5).astype(np.float32)

        # get the best validated model
        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=5,
            var_smoothing=var_smoothing,
            grid_search=True)

        # print all possible parameter values and the best parameters
        gnb.print_parameter_candidates()
        gnb.print_best_estimator()

        # return the accuracy score
        return gnb.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

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
#         hidden_layer_sizes = reciprocal_distrobution_hls.rvs(size=10).astype(np.int)
#         max_iter = reciprocal_distribution_mi.rvs(size=10).astype(np.int)
        # best result over all hidden_layer_sizes: 442
        # best result over all max_iter: 1222

        # scale down parameters around its best result
        np.random.seed(0)
        hidden_layer_sizes = norm.rvs(loc=442, scale=10, size=2).astype(np.int)
        max_iter = norm.rvs(loc=1222, scale=50, size=2).astype(np.int)

        # get the best random validated model
        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_search=True)

        # print all possible parameter values and best parameters
        nnc.print_parameter_candidates()
        nnc.print_best_estimator()

        # return the accuracy score
        return nnc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)


if __name__ == '__main__':
    seb = Seismic_bumps()
    print("accuracy on the actual test set:")
    print('KNN: %.2f %%' % (seb.k_nearest_neighbours() * 100))
    print('SVC: %.2f %%' % (seb.support_vector_classifier() * 100))
    print('DTC: %.2f %%' % (seb.decision_tree_classifier() * 100))
    print('RFC: %.2f %%' % (seb.random_forest_classifier() * 100))
    print('ABC: %.2f %%' % (seb.ada_boost_classifier() * 100))
    print(' LR: %.2f %%' % (seb.logistic_regression() * 100))
    print('GNB: %.2f %%' % (seb.gaussian_naive_bayes() * 100))
    print('NNC: %.2f %%' % (seb.neural_network_classifier() * 100))
