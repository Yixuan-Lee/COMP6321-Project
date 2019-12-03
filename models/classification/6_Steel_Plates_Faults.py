import os
import numpy as np
import scipy.stats
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from models import settings
from models.classification.ada_boost_classifier import Ada_boost_classifier
from models.classification.decision_tree_classifier import Decision_tree_classifier
from models.classification.gaussian_naive_bayes import Gaussian_naive_bayes
from models.classification.k_nearest_neighbours import K_nearest_neighbours
from models.classification.logistic_regression import Logistic_regression
from models.classification.neural_network_classifier import Neural_network_classifier
from models.classification.random_forest_classifier import Random_forest_classifier
from models.classification.support_vector_classifier import Support_vector_classifier
import warnings




class Steel_Plates_Faults:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/6_Steel_Plates_Faults'
        filename = 'Faults.NNA'

        # read data from the source file
        self.data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
                                            filename), dtype=np.float64)
        self.targets = self.data[:, -7:]
        self.targets = np.argwhere(self.targets == 1)[:, -1]  # transformation of the targets matrix
        self.data = self.data[:, :-7]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                             random_state=0)

        # normalize the training set
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)

    def print_self(self):
        print(self.data.shape)
        print(self.targets)

    def k_nearest_neighbours(self):
        # try n_neighbors from 1 to 100
        knn = K_nearest_neighbours(x_train=self.x_train,
                                   y_train=self.y_train,
                                   cv=3,
                                   # n_iter=30,
                                   n_neighbors=np.arange(1, 100, 1),
                                   grid_search=True,
                                   n_jobs=10)
        knn.print_parameter_candidates()
        knn.print_best_estimator()
        return (knn.evaluate(data=self.x_train, targets=self.y_train),
                knn.evaluate(data=self.x_test, targets=self.y_test))

    def support_vector_classifier(self):
        kernel_distribution = ['linear', 'rbf', 'sigmoid']
        # C = np.logspace(-3,3,num=7)
        # gamma = np.logspace(-3,3,num=7)
        # svm = Support_vector_classifier(x_train= self.x_train,
        #                                 y_train= self.y_train,
        #                                 cv=3,
        #                                 # n_iter=30,
        #                                 kernel= kernel_distribution,
        #                                 C = C,
        #                                 gamma= gamma,
        #                                 grid_search= True,
        #                                 n_jobs=10)
        # try C from 1 to 1000 with reciprocal distribution
        # try gamma from 0.01 to 10 with reciprocal distribution
        # Best estimator: SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
        #                decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
        #                max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
        #                verbose=False)
        # SVC: 76.60 %
        np.random.seed(0)
        C = scipy.stats.norm(10,10).rvs(1000)
        C = C[C > 0]
        gamma = scipy.stats.norm(0.01,0.01).rvs(1000)
        gamma = gamma[gamma>0]
        svm = Support_vector_classifier(x_train= self.x_train,
                                        y_train= self.y_train,
                                        cv=3,
                                        n_iter=500,
                                        kernel= ('rbf',),
                                        C = C,
                                        gamma= gamma,
                                        random_search= True,
                                        n_jobs=10)
        svm.print_parameter_candidates()
        svm.print_best_estimator()
        return (svm.evaluate(data=self.x_train, targets=self.y_train),
                svm.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_classifier(self):
        # try max_depth form 1 to 50
        dtc = Decision_tree_classifier(x_train=self.x_train,
                                       y_train=self.y_train,
                                       cv=3,
                                       n_iter=30,
                                       # max_depth=np.linspace(1, 50, 50, dtype=np.int, endpoint=True),
                                       # min_samples_leaf=np.linspace(1, 10, 10, dtype=np.int, endpoint=True),
                                       grid_search=True)
        dtc.print_parameter_candidates()
        dtc.print_best_estimator()
        return (dtc.evaluate(data=self.x_train, targets=self.y_train),
                dtc.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_classifier(self):
        # try max_depth from 1 to 50
        # try n_estimators from 1 to 50
        rfc = Random_forest_classifier(x_train=self.x_train,
                                       y_train=self.y_train,
                                       cv=3,
                                       n_iter=30,
                                       max_depth=np.linspace(1, 20, 20, dtype=np.int, endpoint=True),
                                       n_estimators=np.linspace(1, 50, 50, dtype=np.int, endpoint=True),
                                       grid_search=True,
                                       n_jobs=20)
        rfc.print_parameter_candidates()
        rfc.print_best_estimator()
        return (rfc.evaluate(data=self.x_train, targets=self.y_train),
                rfc.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_classifier(self):
        # lr = np.logspace(-3, 3, num=7)
        # abc = Ada_boost_classifier(x_train=self.x_train,
        #                            y_train=self.y_train,
        #                            cv=3,
        #                            n_iter=30,
        #                            n_estimators=np.linspace(1, 50, 25, dtype=np.int, endpoint=True),
        #                            learning_rate=lr,
        #                            n_jobs= 10,
        #                            grid_search = True)
        # for the algorithm of Adaboost, as what sklearn mentioned:
        #   "The SAMME.R algorithm typically converges faster than SAMME,
        #   achieving a lower test error with fewer boosting iterations."
        #  Here we only try to find the best n_estimators for training data
        #  I added the learning_rata but the result could be worse or the same as before!!!!
        #  Best estimator :  AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
        #                    n_estimators=3, random_state=0)
        # ABC: 53.35%
        np.random.seed(0)
        lr = scipy.stats.norm(1,.5).rvs(100)
        lr = lr[lr>0]
        abc = Ada_boost_classifier(x_train=self.x_train,
                                   y_train=self.y_train,
                                   cv=3,
                                   n_iter=100,
                                   # n_estimators=np.linspace(1, 50, 50, dtype=np.int, endpoint=True),
                                   # learning_rate=lr,
                                   random_search=True)
        abc.print_parameter_candidates()
        abc.print_best_estimator()
        return (abc.evaluate(data=self.x_train, targets=self.y_train),
                abc.evaluate(data=self.x_test, targets=self.y_test))

    def logistic_regression(self):
        # C = np.logspace(-3,3,num=7)
        # lr = Logistic_regression(x_train=self.x_train,
        #                          y_train=self.y_train,
        #                          cv=3,
        #                          n_iter=30,
        #                          C=C,
        #                          # penalty= ('l1','l2'),
        #                          max_iter=np.linspace(10,1000,10),
        #                          grid_search=True,
        #                          n_jobs=10
        #                          )
        #  According to Scikit Documentation: The SAGA solver is often the best choice.
        #  for dual parameter : Prefer dual=False when n_samples > n_features.
        #  for penalty parameter :  ‘elasticnet’ is only supported by the ‘saga’ solver.
        #  result does not change too much !!!
        # Best estimator: LogisticRegression(C=100.0, class_weight=None, dual=False, fit_intercept=True,
        #                               intercept_scaling=1, l1_ratio=None, max_iter=120.0,
        #                               multi_class='warn', n_jobs=None, penalty='l2',
        #                               random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
        #                               warm_start=False)
        # LR: 72.23 %
        np.random.seed(0)
        C = scipy.stats.norm(100,10).rvs(200)
        C = C[C>0]
        lr = Logistic_regression(x_train=self.x_train,
                                 y_train=self.y_train,
                                 cv=3,
                                 n_iter=100,
                                 # C=C,
                                 # penalty= ('l1','l2'),
                                 # max_iter=[1000],
                                 random_search=True,
                                 n_jobs=10
                                 )
        lr.print_parameter_candidates()
        lr.print_best_estimator()
        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_naive_bayes(self):
        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            var_smoothing=np.logspace(start=-9, stop=-6, base=10, num=4, dtype=np.float32),
            grid_search=True)

        # print all possible parameter values and the best parameters
        gnb.print_parameter_candidates()
        gnb.print_best_estimator()

        # return the accuracy score

        return (gnb.evaluate(data=self.x_train, targets=self.y_train),
                gnb.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_classifier(self):
        # get the best random validated model
        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            # hidden_layer_sizes=np.arange(100, 1000),
            # max_iter=np.arange(1000, 10000),
            random_search=True)

        # print all possible parameter values and best parameters
        nnc.print_parameter_candidates()
        nnc.print_best_estimator()

        # return the accuracy score

        # return the accuracy score
        return (nnc.evaluate(data=self.x_train, targets=self.y_train),
                nnc.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignore sklearn deprecation warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore sklearn deprecation warnings
    spf = Steel_Plates_Faults()
    np.random.seed(0)
    # spf.print_self()
    print('KNN: %.2f%%' % (spf.k_nearest_neighbours()*100))
    print('SVC: %.2f%%' % (spf.support_vector_classifier()*100))
    print('DTC: %.2f%%' % (spf.decision_tree_classifier()*100))
    print('RFC: %.2f%%' % (spf.random_forest_classifier()*100))
    print('ABC: %.2f%%' % (spf.ada_boost_classifier()*100))
    print(' LR: %.2f%%' % (spf.logistic_regression()*100))
    print('GNB: %.2f%%' % (spf.gaussian_naive_bayes()*100))
    print('NNC: %.2f%%' % (spf.neural_network_classifier()*100))
