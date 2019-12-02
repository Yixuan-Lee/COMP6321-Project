import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
import scipy.stats
from scipy.io import arff
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

class Thoracic_Surgery_Data:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/9_Thoracic_Surgery_Data/'
        filename = 'ThoraricSurgery.arff'

        # read the data file
        file, meta = arff.loadarff(os.path.join(filepath, filename))
        self.data = np.asarray(file.tolist())
        self.targets = self.data[:, -1]
        self.data = np.delete(self.data,-1,axis=1)
        numeric1 = self.data[:,-1]
        numeric2 = self.data[:,1]
        numeric3 = self.data[:,2]
        self.data = np.delete(self.data,-1,axis=1)
        self.data = np.delete(self.data,1,axis=1)
        self.data = np.delete(self.data,1,axis=1)
        encode = preprocessing.OneHotEncoder().fit(self.data)
        self.data = encode.transform(self.data).toarray()
        self.data = np.insert(self.data, 0, values=numeric1, axis=1)
        self.data = np.insert(self.data, 0, values=numeric2, axis=1)
        self.data = np.insert(self.data, 0, values=numeric3, axis=1)
        self.data = np.asarray(self.data, dtype=np.float32)

        # split into the train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)
        print(self.x_train)

    def k_nearest_neighbours(self):
        n_neighbors = range(1, 100, 1)  # [1, 3, 5, ..., 99]
        knn = K_nearest_neighbours(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            n_neighbors=n_neighbors,
            grid_search=True)

        knn.print_parameter_candidates()
        knn.print_best_estimator()

        return knn.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def support_vector_classifier(self):
        C = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3, dtype=np.float32)  # [0.1, 1, 10]
        kernel = ['linear', 'rbf', 'sigmoid']

        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            C=C,
            kernel=kernel,
            gamma=gamma,
            random_search=True)

        svc.print_parameter_candidates()
        svc.print_best_estimator()

        return svc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_classifier(self):
        criterion = ['gini', 'entropy']
        max_depth = range(1, 100, 2)

        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=6,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        dtc.print_parameter_candidates()
        dtc.print_best_estimator()

        return dtc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_classifier(self):
        criterion = ['gini', 'entropy']
        max_depth = range(1, 20, 2)

        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=6,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        rfc.print_parameter_candidates()
        rfc.print_best_estimator()

        return rfc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def ada_boost_classifier(self):
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)

        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=6,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

        abc.print_parameter_candidates()
        abc.print_best_estimator()

        return abc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def logistic_regression(self):
        C = np.logspace(start=-4, stop=4, base=10, num=9, dtype=np.float32)

        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            C=C,
            random_search=True)

        lr.print_parameter_candidates()
        lr.print_best_estimator()

        return lr.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_naive_bayes(self):
        var_smoothing = np.logspace(start=-9, stop=-6, base=10, num=4,
            dtype=np.float32)

        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=6,
            var_smoothing=var_smoothing,
            grid_search=True)

        gnb.print_parameter_candidates()
        gnb.print_best_estimator()

        return gnb.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def neural_network_classifier(self):
        reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        hidden_layer_sizes = \
            reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)

        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=6,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_search=True)

        nnc.print_parameter_candidates()
        nnc.print_best_estimator()

        return nnc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)


if __name__ == '__main__':
    dr = Thoracic_Surgery_Data()
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