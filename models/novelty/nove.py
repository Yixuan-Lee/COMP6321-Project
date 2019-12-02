
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
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

        # split into the train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set and the testing set
        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    ##################### Model training #####################


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
            class_weight=('balanced',),
            # cv=5,
            # n_jobs=3,
            # C=C,
            # gamma=gamma,
            # kernel=kernel,
            # grid_search=True
        )

        # print all possible parameter values and the best parameters
        svc.print_parameter_candidates()
        svc.print_best_estimator()
        print('svc precision: %.2f %%' % (svc.precision(self.x_test,self.y_test) * 100))
        print('svc recall: %.2f %%' % (svc.recall(self.x_test,self.y_test) * 100))
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
            class_weight=({1:8,0:2},),
            grid_search=True)

        # print all possible parameter values and the best parameters
        dtc.print_parameter_candidates()
        dtc.print_best_estimator()

        print('dtc precision: %.2f %%' % (dtc.precision(self.x_test, self.y_test) * 100))
        print('dtc recall: %.2f %%' % (dtc.recall(self.x_test, self.y_test) * 100))
        # return the accuracy score
        return dtc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

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
            grid_search=True,
            class_weight=({1:8,0:2},)
        )

        # print all possible parameter values and the best parameters
        rfc.print_parameter_candidates()
        rfc.print_best_estimator()

        print('rfc precision: %.2f %%' % (rfc.precision(self.x_test, self.y_test) * 100))
        print('rfc recall: %.2f %%' % (rfc.recall(self.x_test, self.y_test) * 100))
        # return the accuracy score
        return rfc.accuracy_score(
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
            grid_search=True,
            class_weight=({1:8,0:2},)
        )

        # print all possible parameter values and the best parameters
        lr.print_parameter_candidates()
        lr.print_best_estimator()

        print('lr precision: %.2f %%' % (lr.precision(self.x_test, self.y_test) * 100))
        print('lr recall: %.2f %%' % (lr.recall(self.x_test, self.y_test) * 100))

        # return the accuracy score
        return lr.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)







if __name__ == '__main__':
    doccc = Default_of_credit_card_clients()
    #print("accuracy on the actual test set:")
    print('SVC: %.2f %%' % (doccc.support_vector_classifier() * 100))
    #print('DTC: %.2f %%' % (doccc.decision_tree_classifier() * 100))
    #print('RFC: %.2f %%' % (doccc.random_forest_classifier() * 100))
    #print(' LR: %.2f %%' % (doccc.logistic_regression() * 100))

