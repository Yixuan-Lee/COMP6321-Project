import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
import sklearn.model_selection  # for RandomizedSearchCV
from models import settings     # for retrieving root path
from scipy.io import arff       # for loading .arff file
from sklearn.neighbors import KNeighborsClassifier
# from k_nearest_neighbours import K_nearest_neighbours
from sklearn.svm import SVC
# from support_vector_classifier import Support_vector_classifier
from sklearn.tree import DecisionTreeClassifier
# from decision_tree_classifier import Decision_tree_classifier
from sklearn.ensemble import RandomForestClassifier
# from random_forest_classifier import Random_forest_classifier
from sklearn.ensemble import AdaBoostClassifier
# from ada_boost_classifier import Ada_boost_classifier
from sklearn.linear_model import LogisticRegression
# from logistic_regression import Logistic_regression

from gaussian_naive_bayes import Gaussian_naive_bayes
from sklearn.neural_network import MLPClassifier
from neural_network_classifier import Neural_network_classifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

        # normalize the training set
        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)

    ##################### Model training #####################
    def k_nearest_neighbours(self):
        """
        for knn, i train on the training data using different :
            1) n_neighbors,
            2) weights
            3) p
        since these 3 parameters vary in small range, so i choose to
        use GridSearchCV

        :return: test accuracy of the knn best model
        """

        # Grid Search Validation
        # define param_grid argument to give GridSearchCV
        n_neighbors = range(1, 100, 2)  # [1, 3, 5, ..., 99]
        weights = ['uniform', 'distance']
        p = [1, 2]
        param_grid = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_params_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy

    def support_vector_classifier(self):
        """
        for svc, i train on the training data using different :
            2) C
            3) gamma
        :return: test accuracy of the svc best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        C = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3, dtype=np.float32)  # [0.01, 0.1, 1, 10]
        param_grid = {
            'C': C,
            'gamma': gamma,
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=SVC(kernel='rbf', random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy

    def decision_tree_classifier(self):
        """
        for dtc, i train on the training data using different :
            1) criterion
            2) max_depth

        :return: test accuracy of the dtc best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        criterion = ['gini', 'entropy']
        max_depth = range(1, 100, 2)
        param_grid = {
            'criterion': criterion,
            'max_depth': max_depth
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy

    def random_forest_classifier(self):
        """
        for rfc, i train on the training data using different :
            1) criterion
            2) max_depth

        :return: test accuracy of the dtc best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        criterion = ['gini', 'entropy']
        max_depth = range(1, 20, 2)
        param_grid = {
            'criterion': criterion,
            'max_depth': max_depth
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=100, random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy


    def ada_boost_classifier(self):
        """
        for abc, i train on the training data using different :
            1) n_estimators
            2) learning_rate

        :return: test accuracy of the dtc best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)
        param_grid = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=AdaBoostClassifier(algorithm='SAMME', random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy

    def logistic_regression(self):
        """
        for lr, i train on the training data using different :
            1) C

        :return: test accuracy of the dtc best model
        """

        # GridSearch Cross Validation
        # define param_grid argument to give GridSearchCV
        C = np.logspace(start=-4, stop=4, base=10, num=9, dtype=np.float32)
        param_grid = {
            'C': C
        }

        # fit a 5-fold GridSearchCV instance
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=LogisticRegression(max_iter=10000, solver='lbfgs',
                random_state=0),
            param_grid=param_grid,
            cv=5)
        gscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_grid)
        print(gscv.best_estimator_)

        # estimate on the test set
        best_gscv = gscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_gscv.predict(self.x_test))

        return test_accuracy

    def gaussian_naive_bayes(self):
        """
        for gnb, i train on the training data using different :
            1)
            2)

        :return: test accuracy of the gnb best model
        """
        gnb = Gaussian_naive_bayes()
        gnb.train(self.x_train, self.y_train)
        return gnb.get_accuracy(self.x_test, self.y_test)

    def neural_network_classifier(self):
        """
        for nnc, i train on the training data using different :
            1) hidden_layer_sizes
            2) max_iter

        :return: test accuracy of the nnr best model
        """
        # RandomSearch Cross Validation
        # define param_dist argument to give RandomSearchCV
        reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        reci_hidden_layer_sizes = \
            reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)
        param_dist = {
            'hidden_layer_sizes': reci_hidden_layer_sizes,
            'max_iter': max_iter
        }

        # fit a 5-fold RandomSearchCV instance
        rscv = sklearn.model_selection.RandomizedSearchCV(
            estimator=MLPClassifier(activation='relu', solver='lbfgs',
                alpha=1e-5, random_state=0),
            param_distributions=param_dist,
            verbose=1,
            cv=5,
            n_jobs=5,  # 5 concurrent workers
            n_iter=5,
            random_state=0)
        rscv.fit(self.x_train, self.y_train)

        # print the candidates and best parameters
        print('candidates: ', param_dist)
        print(rscv.best_estimator_)

        # estimate on the test set
        best_rscv = rscv.best_estimator_
        test_accuracy = accuracy_score(
            y_true=self.y_test,
            y_pred=best_rscv.predict(self.x_test))

        return test_accuracy


if __name__ == '__main__':
    dr = Diabetic_retinopathy()
    print("accuracy on the actual test set:")
    # print('KNN: %.2f %%' % (dr.k_nearest_neighbours() * 100))
    # print('SVC: %.2f %%' % (dr.support_vector_classifier() * 100))
    # print('DTC: %.2f %%' % (dr.decision_tree_classifier() * 100))
    # print('RFC: %.2f %%' % (dr.random_forest_classifier() * 100))
    # print('ABC: %.2f %%' % (dr.ada_boost_classifier() * 100))
    # print(' LR: %.2f %%' % (dr.logistic_regression() * 100))

    # print('GNB: %.2f %%' % (dr.gaussian_naive_bayes() * 100))

    print('NNC: %.2f %%' % (dr.neural_network_classifier() * 100))
