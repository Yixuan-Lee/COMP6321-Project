import os
import numpy as np
import pandas as pd
from models import settings
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

        # normalize the training set
        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)

    ##################### Model training #####################
    def k_nearest_neighbours(self):
        knn = K_nearest_neighbours()
        knn.train(self.x_train, self.y_train)
        return knn.get_accuracy(self.x_test, self.y_test)

    def support_vector_classifier(self):
        svc = Support_vector_classifier()
        svc.train(self.x_train, self.y_train)
        return svc.get_accuracy(self.x_test, self.y_test)

    def decision_tree_classifier(self):
        dtc = Decision_tree_classifier()
        dtc.train(self.x_train, self.y_train)
        return dtc.get_accuracy(self.x_test, self.y_test)

    def random_forest_classifier(self):
        rfc = Random_forest_classifier()
        rfc.train(self.x_train, self.y_train)
        return rfc.get_accuracy(self.x_test, self.y_test)

    def ada_boost_classifier(self):
        abc = Ada_boost_classifier()
        abc.train(self.x_train, self.y_train)
        return abc.get_accuracy(self.x_test, self.y_test)

    def logistic_regression(self):
        lr = Logistic_regression()
        lr.train(self.x_train, self.y_train)
        return lr.get_accuracy(self.x_test, self.y_test)

    def gaussian_naive_bayes(self):
        gnb = Gaussian_naive_bayes()
        gnb.train(self.x_train, self.y_train)
        return gnb.get_accuracy(self.x_test, self.y_test)

    def neural_network_classifier(self):
        nnc = Neural_network_classifier(hls=(15,), s='lbfgs', alp=1e-5)
        nnc.train(self.x_train, self.y_train)
        return nnc.get_accuracy(self.x_test, self.y_test)

if __name__ == '__main__':
    doccc = Default_of_credit_card_clients()
    print("accuracy on the actual test set:")
    print('KNN: %.2f %%' % (doccc.k_nearest_neighbours() * 100))
    print('SVC: %.2f %%' % (doccc.support_vector_classifier() * 100))
    print('DTC: %.2f %%' % (doccc.decision_tree_classifier() * 100))
    print('RFC: %.2f %%' % (doccc.random_forest_classifier() * 100))
    print('ABC: %.2f %%' % (doccc.ada_boost_classifier() * 100))
    print(' LR: %.2f %%' % (doccc.logistic_regression() * 100))
    print('GNB: %.2f %%' % (doccc.gaussian_naive_bayes() * 100))
    print('NNC: %.2f %%' % (doccc.neural_network_classifier() * 100))
