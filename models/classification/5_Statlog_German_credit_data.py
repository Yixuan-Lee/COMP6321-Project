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



class Statlog_German_Credit:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/classification_datasets/5_Statlog_German_credit_data'
        filename = 'german.data-numeric'

        # read data from the source file
        self.data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
                                            filename), dtype=np.int)
        self.targets = self.data[:, -1]
        self.data = self.data[:, :-1]

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
        print(self.data)
        print(self.targets)

    def k_nearest_neighbours(self):
        # try n_neighbors from 1 to 100
        knn = K_nearest_neighbours(x_train= self.x_train,
                                   y_train= self.y_train,
                                   cv=3,
                                   n_iter=30,
                                   n_neighbors=np.arange(1, 100, 1),
                                   grid_search= True,
                                   n_jobs=10)
        knn.print_parameter_candidates()
        knn.print_best_estimator()
        return knn.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

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
        # Best estimator: SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
        #                decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
        #                max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
        #                verbose=False)
        # try C from 1 to 1000 with reciprocal distribution
        # try gamma from 0.01 to 10 with reciprocal distribution
        C = scipy.stats.norm(10,10).rvs(100)
        C = C[C > 0]
        gamma = scipy.stats.norm(0.001,0.001).rvs(100)
        gamma = gamma[gamma>0]
        svm = Support_vector_classifier(x_train= self.x_train,
                                        y_train= self.y_train,
                                        cv=3,
                                        n_iter=100,
                                        kernel= ('rbf',),
                                        C = C,
                                        gamma= gamma,
                                        random_search= True,
                                        n_jobs=10)
        svm.print_parameter_candidates()
        svm.print_best_estimator()
        return svm.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def decision_tree_classifier(self):
        # try max_depth form 1 to 50
        dtc = Decision_tree_classifier(x_train= self.x_train,
                                       y_train= self.y_train,
                                       cv=3,
                                       n_iter=30,
                                       max_depth= np.linspace(1,50,50,dtype=np.int,endpoint=True),
                                       min_samples_leaf= np.linspace(1,10,10,dtype=np.int,endpoint=True),
                                       grid_search=True,
                                       n_jobs=10)
        dtc.print_parameter_candidates()
        dtc.print_best_estimator()
        return dtc.accuracy_score(self.x_test, self.y_test)

    def random_forest_classifier(self):
        # try max_depth from 1 to 50
        # try n_estimators from 1 to 50
        rfc = Random_forest_classifier(x_train= self.x_train,
                                       y_train= self.y_train,
                                       cv=3,
                                       n_iter=30,
                                       # max_depth=np.linspace(1, 50, 50, dtype=np.int, endpoint=True),
                                       # n_estimators=np.linspace(1, 50, 50, dtype=np.int, endpoint=True),
                                       grid_search= True,
                                       n_jobs=10)
        rfc.print_parameter_candidates()
        rfc.print_best_estimator()
        return rfc.accuracy_score(self.x_test, self.y_test)

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
        # Best
        # estimator: AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
        #                               n_estimators=23, random_state=0)
        # ABC: 74.85 %
        lr = scipy.stats.norm(1,.5).rvs(100)
        lr = lr[lr>0]
        abc = Ada_boost_classifier(x_train=self.x_train,
                                   y_train=self.y_train,
                                   cv=3,
                                   n_iter=100,
                                   n_estimators=[23],
                                   learning_rate=lr,
                                   n_jobs= 10,
                                   random_search = True)
        abc.print_parameter_candidates()
        abc.print_best_estimator()
        return abc.accuracy_score(self.x_test, self.y_test)

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
        # Best estimator: LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
        #                               intercept_scaling=1, l1_ratio=None, max_iter=10.0,
        #                               multi_class='warn', n_jobs=None, penalty='l2',
        #                               random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
        #                               warm_start=False)
        # LR: 76.36 %
        C = scipy.stats.norm(.1,.1).rvs(200)
        C = C[C>0]
        lr = Logistic_regression(x_train=self.x_train,
                                 y_train=self.y_train,
                                 cv=3,
                                 n_iter=100,
                                 C = C,
                                 # penalty= ('l1','l2'),
                                 max_iter= [1000],
                                 random_search= True,
                                 n_jobs=10
                                 )
        lr.print_parameter_candidates()
        lr.print_best_estimator()
        return lr.accuracy_score(self.x_test, self.y_test)

    def gaussian_naive_bayes(self):
        gnb = Gaussian_naive_bayes(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            var_smoothing=np.logspace(start=-9, stop=-6, base=10, num=4,dtype=np.float32),
            grid_search=True)

        # print all possible parameter values and the best parameters
        gnb.print_parameter_candidates()
        gnb.print_best_estimator()

        # return the accuracy score
        return gnb.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)

    def neural_network_classifier(self):
        activation = ['logistic', 'tanh', 'relu']
        hls = np.logspace(2,4,num=3,dtype=np.int)
        mi = np.logspace(3,5,num=3,dtype=np.int)

        # Best estimator: MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
        #                          beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
        #                          hidden_layer_sizes=100, learning_rate='constant',
        #                          learning_rate_init=0.001, max_iter=1000, momentum=0.9,
        #                          n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
        #                          random_state=0, shuffle=True, solver='adam', tol=0.0001,
        #                          validation_fraction=0.1, verbose=False, warm_start=False)
        # NNC: 75.76 %
        # get the best random validated model
        nnc = Neural_network_classifier(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=100,
            # activation=activation,
            # hidden_layer_sizes=hls,
            # max_iter=mi,
            grid_search=True,
            n_jobs=10)


        # print all possible parameter values and best parameters
        nnc.print_parameter_candidates()
        nnc.print_best_estimator()

        # return the accuracy score
        return nnc.accuracy_score(
            x_test=self.x_test,
            y_test=self.y_test)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignore sklearn deprecation warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore sklearn deprecation warnings
    sgcd = Statlog_German_Credit()
    np.random.seed(0)
    # sgcd.print_self()
    print('KNN: %.2f%%' % (sgcd.k_nearest_neighbours()*100))
    print('SVC: %.2f%%' % (sgcd.support_vector_classifier()*100))
    print('DTC: %.2f%%' % (sgcd.decision_tree_classifier()*100))
    print('RFC: %.2f%%' % (sgcd.random_forest_classifier()*100))
    print('ABC: %.2f%%' % (sgcd.ada_boost_classifier()*100))
    print(' LR: %.2f%%' % (sgcd.logistic_regression()*100))
    print('GNB: %.2f%%' % (sgcd.gaussian_naive_bayes()*100))
    print('NNC: %.2f%%' % (sgcd.neural_network_classifier()*100))
