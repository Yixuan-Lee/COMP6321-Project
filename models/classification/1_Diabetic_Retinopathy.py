import os
import numpy as np
import scipy
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from scipy.io import arff       # For loading .arff file
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
        # define arguments given to GridSearchCV
        n_neighbors = range(1, 100, 2)  # [1, 3, 5, ..., 99]

        # get the best validated model
        knn = K_nearest_neighbours(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_neighbors=n_neighbors,
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
        # define arguments given to GridSearchCV
        C = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)  # [0.1, 1, 10, 100, 1000]
        gamma = np.logspace(start=-1, stop=1, base=10, num=3, dtype=np.float32)  # [0.01, 0.1, 1, 10]
        kernel = ['linear', 'rbf', 'sigmoid']

        # get the best validated model
        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            kernel=kernel,
            gamma=gamma,
            grid_search=True)

        # print all possible parameter values
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

        # define param_grid argument to give GridSearchCV
        criterion = ['gini', 'entropy']
        max_depth = range(1, 100, 2)

        # get the best validated model
        dtc = Decision_tree_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
            2) max_depth

        :return: test accuracy of the dtc best model
        """
        # define arguments given to GridSearchCV
        criterion = ['gini', 'entropy']
        max_depth = range(1, 20, 2)

        # get the best validated model
        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            criterion=criterion,
            max_depth=max_depth,
            grid_search=True)

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
        # define arguments given to GridSearchCV
        n_estimators = range(1, 100, 5)
        learning_rate = np.logspace(start=-2, stop=0, base=10, num=3,
            dtype=np.float32)

        # get the best validated model
        abc = Ada_boost_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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

        :return: test accuracy of the dtc best model
        """
        # define arguments given to GridSearchCV
        C = np.logspace(start=-4, stop=4, base=10, num=9, dtype=np.float32)

        # get the best validated model
        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
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
        # define arguments given to GridSearchCV
        var_smoothing = np.logspace(start=-9, stop=-6, base=10, num=4,
            dtype=np.float32)

        # get the best validated model
        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
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
        # define arguments given to RandomSearchCV
        reciprocal_distrobution_hls = scipy.stats.reciprocal(a=100, b=1000)
        reciprocal_distribution_mi = scipy.stats.reciprocal(a=1000, b=10000)
        np.random.seed(0)
        hidden_layer_sizes = \
            reciprocal_distrobution_hls.rvs(size=5).astype(np.int)
        max_iter = reciprocal_distribution_mi.rvs(size=5).astype(np.int)

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
    dr = Diabetic_retinopathy()
    print("accuracy on the actual test set:")
    print('KNN: %.2f %%' % (dr.k_nearest_neighbours() * 100))
    print('SVC: %.2f %%' % (dr.support_vector_classifier() * 100))
    print('DTC: %.2f %%' % (dr.decision_tree_classifier() * 100))
    print('RFC: %.2f %%' % (dr.random_forest_classifier() * 100))
    print('ABC: %.2f %%' % (dr.ada_boost_classifier() * 100))
    print(' LR: %.2f %%' % (dr.logistic_regression() * 100))
    print('GNB: %.2f %%' % (dr.gaussian_naive_bayes() * 100))
    print('NNC: %.2f %%' % (dr.neural_network_classifier() * 100))
