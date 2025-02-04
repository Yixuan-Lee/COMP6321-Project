import os
import sys
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from pathlib import Path
from k_nearest_neighbours import K_nearest_neighbours
from support_vector_classifier import Support_vector_classifier
from decision_tree_classifier import Decision_tree_classifier
from random_forest_classifier import Random_forest_classifier
from ada_boost_classifier import Ada_boost_classifier
from logistic_regression import Logistic_regression
from gaussian_naive_bayes import Gaussian_naive_bayes
from neural_network_classifier import Neural_network_classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


class Adult:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = './datasets/classification_datasets/7_Adult/'
        filename1 = 'adult.data'
        filename2 = 'adult.test'
        chara_list = [1, 3, 5, 6, 7, 8, 9, 13]
        feature_list = []
        feature_dict = {}

        # separately read the training and test data as file1 and file2
        file1 = pd.read_table(filepath+filename1, sep='\s+')
        self.data = np.asarray(file1)
        for i in range(0,len(self.data[0])-1):
            for j in range(len(self.data[:,i])):
                self.data[:,i][j] = self.data[:,i][j][:len(self.data[:,i][j])-1]
        for i in chara_list:
            for j in self.data[:, i]:
                if j not in feature_list:
                    feature_list.append(j)
                feature_dict[j] = feature_list.index(j)
            feature_list = []
        for i in chara_list:
            for j in range(len(self.data[:,i])):
                self.data[:,i][j] = feature_dict[self.data[:,i][j]]
        self.targets = self.data[:,-1]
        self.data = np.delete(self.data,-1,axis=1)
        self.x_train = self.data
        self.y_train = self.targets

        file2 = pd.read_table(filepath+filename2, sep='\s+',skiprows=1)
        self.data = np.asarray(file2)
        for i in range(0,len(self.data[0])):
            for j in range(len(self.data[:,i])):
                self.data[:,i][j] = self.data[:,i][j][:len(self.data[:,i][j])-1]
        for i in chara_list:
            for j in self.data[:, i]:
                if j not in feature_list:
                    feature_list.append(j)
                feature_dict[j] = feature_list.index(j)
            feature_list = []
        for i in chara_list:
            for j in range(len(self.data[:,i])):
                self.data[:,i][j] = feature_dict[self.data[:,i][j]]
        self.targets = self.data[:,-1]
        self.data = np.delete(self.data,-1,axis=1)
        self.x_test = self.data
        self.y_test = self.targets

    def k_nearest_neighbours(self):
        n_neighbors = range(1, 100)
        knn = K_nearest_neighbours(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_neighbors=n_neighbors,
            random_search=True)

        # knn.print_parameter_candidates()
        # knn.print_best_estimator()

        return (knn.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                knn.evaluate(data=self.x_test, targets=self.y_test, average='micro'))


    def support_vector_classifier(self):
        C = np.logspace(start=-1, stop=-1, base=10, num=5, dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=-1, base=10, num=3, dtype=np.float32)  # [0.01, 0.1, 1, 10]
        kernel = ['linear']

        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            n_jobs=-1,
            kernel=kernel,
            gamma=gamma,
            grid_search=True)

        # svc.print_parameter_candidates()
        # svc.print_best_estimator()

        return (svc.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                svc.evaluate(data=self.x_test, targets=self.y_test, average='micro'))
            
    def decision_tree_classifier(self):
        criterion = ['gini', 'entropy']
        max_depth = range(1, 100, 2)

        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        # dtc.print_parameter_candidates()
        # dtc.print_best_estimator()

        return (dtc.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                dtc.evaluate(data=self.x_test, targets=self.y_test, average='micro'))

    def random_forest_classifier(self):
        criterion = ['gini', 'entropy']
        max_depth = range(1, 20, 2)

        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

        # rfc.print_parameter_candidates()
        # rfc.print_best_estimator()

        return (rfc.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                rfc.evaluate(data=self.x_test, targets=self.y_test, average='micro'))

    def ada_boost_classifier(self):
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)

        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            grid_search=True)

        # abc.print_parameter_candidates()
        # abc.print_best_estimator()

        return (abc.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                abc.evaluate(data=self.x_test, targets=self.y_test, average='micro'))

    def logistic_regression(self):
        C = np.logspace(start=-4, stop=4, base=10, num=9, dtype=np.float32)

        lr = Logistic_regression(
            x_train=self.x_train,
            
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            C=C,
            grid_search=True)

        # lr.print_parameter_candidates()
        # lr.print_best_estimator()

        # return the accuracy score
        return (lr.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                lr.evaluate(data=self.x_test, targets=self.y_test, average='micro'))

    def gaussian_naive_bayes(self):
        var_smoothing = np.logspace(start=-9, stop=-6, base=10, num=4,
            dtype=np.float32)

        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            var_smoothing=var_smoothing,
            grid_search=True)

        # gnb.print_parameter_candidates()
        # gnb.print_best_estimator()

        # return the accuracy score
        return (gnb.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                gnb.evaluate(data=self.x_test, targets=self.y_test, average='micro'))

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
            n_jobs=-1,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_search=True)

        # nnc.print_parameter_candidates()
        # nnc.print_best_estimator()

        return (nnc.evaluate(data=self.x_train, targets=self.y_train, average='micro'),
                nnc.evaluate(data=self.x_test, targets=self.y_test, average='micro'))


if __name__ == '__main__':
    ad = Adult()
    # retrieve the results
    knn_results = ad.k_nearest_neighbours()
    svc_results = ad.support_vector_classifier()
    dtc_results = ad.decision_tree_classifier()
    rfr_results = ad.random_forest_classifier()
    abc_results = ad.ada_boost_classifier()
    lr_results = ad.logistic_regression()
    gnb_results = ad.gaussian_naive_bayes()
    nnc_results = ad.neural_network_classifier()

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