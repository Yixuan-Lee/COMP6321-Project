from models import settings
from scipy.io import arff
from k_nearest_neighbours import K_nearest_neighbours
from support_vector_machines import Support_vector_machines
from decision_tree_classifier import Decision_tree_classifier
from random_forest_classifier import Random_forest_classifier
from ada_boost_classifier import Ada_boost_classifier
from logistic_regression import Logistic_regression
from gaussian_naive_bayes import Gaussian_naive_bayes
from neural_network_classifier import Neural_network_classifier
from sklearn.model_selection import train_test_split
import os
import numpy as np


class diabetic_retinopathy:
    data = []
    targets = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/1_Diabetic_Retinopathy' \
                   '/messidor_features.arff'
        file, meta = arff.loadarff(os.path.join(settings.ROOT_DIR, filepath))
        self.data = np.asarray(file.tolist(), dtype=np.float32)
        self.data = self.data[:, :-1]
        self.targets = self.data[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

    def k_nearest_neighbours(self):
        knn = K_nearest_neighbours()
        knn.train(self.x_train, self.y_train)
        return knn.get_accuracy(self.x_test, self.y_test)

    def support_vector_machines(self):
        svm = Support_vector_machines()
        svm.train(self.x_train, self.y_train)
        return svm.get_accuracy(self.x_test, self.y_test)

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
    dr = diabetic_retinopathy()
    print('KNN: %.2f' % dr.k_nearest_neighbours())
    print('SVM: %.2f' % dr.support_vector_machines())
    print('DTC: %.2f' % dr.decision_tree_classifier())
    print('RFC: %.2f' % dr.random_forest_classifier())
    print('ABC: %.2f' % dr.ada_boost_classifier())
    print(' LR: %.2f' % dr.logistic_regression())
    print('GNB: %.2f' % dr.gaussian_naive_bayes())
    print('NNC: %.2f' % dr.neural_network_classifier())


