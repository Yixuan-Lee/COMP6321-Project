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

        #knn.print_parameter_candidates()
        #knn.print_best_estimator()

        return (knn.evaluate(data=self.x_train, targets=self.y_train),
                knn.evaluate(data=self.x_test, targets=self.y_test))

    def support_vector_classifier(self):
        """
        SVM conclusion : SVM svm performance is not good
        :return: best model's accuracy score
        C = [1,10,100,1000]
        gamma = [0.1,0.01,1,100,1000]
        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            grid_search=True)
        best c :100 gamma : 1 kernel : rbf
        """
        np.random.seed(0)
        C = scipy.stats.reciprocal(1, 1000)
        gamma = scipy.stats.reciprocal(0.01, 20)
        coef0 = scipy.stats.uniform(0, 5)

        svc = Support_vector_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            gamma=gamma,
            coef0=coef0,
            random_search=True)

        #svc.print_parameter_candidates()
        #svc.print_best_estimator()

        return (svc.evaluate(data=self.x_train, targets=self.y_train),
                svc.evaluate(data=self.x_test, targets=self.y_test))

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

        #dtc.print_parameter_candidates()
        #dtc.print_best_estimator()

        return (dtc.evaluate(data=self.x_train, targets=self.y_train),
                dtc.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_classifier(self):
        """
        randomforest without max_depth hyperparameter it has a
        better permoramce
        use grid search to find the range for n estimators
        n_estimators = [1,20,50,100]
        rfc = Random_forest_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_estimators=n_estimators,
            grid_search=True)
            best n estimator = 1
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

        #rfc.print_parameter_candidates()
        #rfc.print_best_estimator()

        return (rfc.evaluate(data=self.x_train, targets=self.y_train),
                rfc.evaluate(data=self.x_test, targets=self.y_test))

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

        #abc.print_parameter_candidates()
        #abc.print_best_estimator()

        return (abc.evaluate(data=self.x_train, targets=self.y_train),
                abc.evaluate(data=self.x_test, targets=self.y_test))

    def logistic_regression(self):
        """
        Logistic regression choose liblinear caz the datasets is small,
        handle no penalty
        solver =[‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            solver = solver
            grid_Search=True)
        solver = 'liblinear'
        """
        np.random.seed(0)
        C = scipy.stats.reciprocal(1, 1000)

        lr = Logistic_regression(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            random_search=True)

        #lr.print_parameter_candidates()
        #lr.print_best_estimator()

        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_naive_bayes(self):
        priors = [(1,), (20,), (50,), (100,)]

        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            var_smoothing=np.logspace(start=-9, stop=-6, base=10, num=4, dtype=np.float32),
            grid_search=True)

        # print all possible parameter values and the best parameters
        #gnb.print_parameter_candidates()
        #gnb.print_best_estimator()

        # return the accuracy score
        return (gnb.evaluate(data=self.x_train, targets=self.y_train),
                gnb.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_classifier(self):
        '''
       nerual network choose from between two hidden layer and one hidden layer,here one hidden layer has the best performance
       hidden_layer_sizes = []
       for i in range(3, 40):
            hidden_layer_sizes.append((i,))
       hidden_layer_sizes = np.asarray(hidden_layer_sizes)
       mlp = Neural_network_classifier(
           x_train=self.x_train,
           y_train=self.y_train,
           activation=('tanh',),
           hidden_layer_sizes=hidden_layer_sizes,
           cv=3,
           n_iter=100,
           grid_search=True,
       )
        '''
        hidden_layer_sizes = []
        for i in range(3, 40):
            for j in range(3, 40):
                hidden_layer_sizes.append((i, j))
        hidden_layer_sizes = np.asarray(hidden_layer_sizes)
        mlp = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            activation=('tanh',),
            hidden_layer_sizes=hidden_layer_sizes,
            cv=3,
            n_iter=100,
            n_jobs=10,
            random_search=True,
        )

        #mlp.print_best_estimator()

        return (mlp.evaluate(data=self.x_train, targets=self.y_train),
                mlp.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    aca = Australian_credit_approval()


    # retrieve the results
    #knn_results = aca.k_nearest_neighbours()
    svc_results = aca.support_vector_classifier()
    #dtc_results = aca.decision_tree_classifier()
    #rfr_results = aca.random_forest_classifier()
    #abc_results = aca.ada_boost_classifier()
    #lr_results = aca.logistic_regression()
    #gnb_results = aca.gaussian_naive_bayes()
    #nnc_results = aca.neural_network_classifier()

    print("(accuracy, recall, prediction) on training set:")
    #print('KNN: (%.3f, %.3f, %.3f)' % (knn_results[0]))
    print('SVC: (%.3f, %.3f, %.3f)' % (svc_results[0]))
    #print('DTC: (%.3f, %.3f, %.3f)' % (dtc_results[0]))
    #print('RFC: (%.3f, %.3f, %.3f)' % (rfr_results[0]))
    #print('ABC: (%.3f, %.3f, %.3f)' % (abc_results[0]))
    #print(' LR: (%.3f, %.3f, %.3f)' % (lr_results[0]))
    #print('GNB: (%.3f, %.3f, %.3f)' % (gnb_results[0]))
    #print('NNC: (%.3f, %.3f, %.3f)' % (nnc_results[0]))

    print("(accuracy, recall, prediction) on testing set:")
    #print('KNN: (%.3f, %.3f, %.3f)' % (knn_results[1]))
    print('SVC: (%.3f, %.3f, %.3f)' % (svc_results[1]))
    #print('DTC: (%.3f, %.3f, %.3f)' % (dtc_results[1]))
    #print('RFC: (%.3f, %.3f, %.3f)' % (rfr_results[1]))
    #print('ABC: (%.3f, %.3f, %.3f)' % (abc_results[1]))
    #print(' LR: (%.3f, %.3f, %.3f)' % (lr_results[1]))
    #print('GNB: (%.3f, %.3f, %.3f)' % (gnb_results[1]))
    #print('NNC: (%.3f, %.3f, %.3f)' % (nnc_results[1]))

