import os
import scipy
import numpy as np
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from k_nearest_neighbours import K_nearest_neighbours
from support_vector_classifier import Support_vector_classifier
from decision_tree_classifier import Decision_tree_classifier
from random_forest_classifier import Random_forest_classifier
from ada_boost_classifier import Ada_boost_classifier
from logistic_regression import Logistic_regression
from gaussian_naive_bayes import Gaussian_naive_bayes
from neural_network_classifier import Neural_network_classifier
from sklearn.model_selection import train_test_split


class Australian_credit_approval:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/4_Statlog_Australian_credit_approval'
        filename = 'australian.dat'

        f = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename),
            delimiter=' ')
        self.data = np.asarray(f[:, :14])
        self.targets = f[:, 14:].reshape(-1)

        print(self.data)
        print(self.targets)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

    def k_nearest_neighbours(self):
        weights = ['uniform', 'distance']
        n_neighbors = range(3, 15)

        knn = K_nearest_neighbours(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_neighbors=n_neighbors,
            weights=weights,
            grid_search=True)

        knn.print_parameter_candidates()
        knn.print_best_estimator()

        return knn.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def support_vector_classifier(self):
        """
        SVM conclusion : SVM svm performance is not good
        :return: best model's accuracy score
        """
        kernel = ('linear', 'rbf', 'sigmoid')
        C = scipy.stats.reciprocal(1, 1000)
        gamma = scipy.stats.reciprocal(0.01, 20)
        coef0 = scipy.stats.uniform(0, 5)

        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            random_search=True)

        svc.print_parameter_candidates()
        svc.print_best_estimator()

        return svc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_classifier(self):
        max_depth = range(1, 14)
        min_samples_leaf = range(1, 9)

        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            grid_search=True)

        dtc.print_parameter_candidates()
        dtc.print_best_estimator()

        return dtc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def random_forest_classifier(self):
        """
        randomforest without max_depth hyperparameter it has a
        better permoramce
        :return:
        """
        n_estimators = range(1, 12)
        max_depth = range(1, 14)

        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            grid_search=True)

        rfc.print_parameter_candidates()
        rfc.print_best_estimator()

        return rfc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def ada_boost_classifier(self):
        n_estimators = range(1, 20)
        algorithm = ('SAMME', 'SAMME.R')

        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_estimators=n_estimators,
            algorithm=algorithm,
            grid_search=True)

        abc.print_parameter_candidates()
        abc.print_best_estimator()

        return abc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def logistic_regression(self):
        """
        Logistic regression choose liblinear caz the datasets is small,
        handle no penalty
        :return:
        """
        C = scipy.stats.reciprocal(1, 1000)

        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            random_search=True)

        lr.print_parameter_candidates()
        lr.print_best_estimator()

        return lr.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def gaussian_naive_bayes(self):
        pass

    def neural_network_classifier(self):
        pass


if __name__ == '__main__':
    aca = Australian_credit_approval()
    print("accuracy on the actual test set:")
    print('KNN: %.2f %%' % (aca.k_nearest_neighbours() * 100))
    print('SVC: %.2f %%' % (aca.support_vector_classifier() * 100))
    print('DTC: %.2f %%' % (aca.decision_tree_classifier() * 100))
    print('RFC: %.2f %%' % (aca.random_forest_classifier() * 100))
    print('ABC: %.2f %%' % (aca.ada_boost_classifier() * 100))
    print(' LR: %.2f %%' % (aca.logistic_regression() * 100))
    print('GNB: %.2f %%' % (aca.gaussian_naive_bayes() * 100))
    print('NNC: %.2f %%' % (aca.neural_network_classifier() * 100))

