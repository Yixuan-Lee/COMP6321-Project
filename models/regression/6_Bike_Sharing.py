import os
import numpy as np
import scipy.stats
import sklearn
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.model_selection import train_test_split

from models import settings
from models.regression.ada_boost_regressor import Ada_boost_regressor
from models.regression.decision_tree_regressor import Decision_tree_regressor
from models.regression.gaussian_process_regressor import Gaussian_process_regressor

from models.regression.linear_least_squares import Linear_least_squares
from models.regression.neural_network_regressor import Neural_network_regressor
from models.regression.random_forest_regressor import Random_forest_regressor
from models.regression.support_vector_regressor import Support_vector_regressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Bike_Sharing:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        np.set_printoptions(threshold=np.inf)
        filepath = 'datasets/regression_datasets/6_Bike_Sharing'
        filename = 'hour.csv'

        # read data from the source file
        self.data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath,
                                            filename), delimiter=',', usecols=range(2, 17), skiprows=1)
        self.targets = self.data[:, -1]
        self.data = self.data[:, :-1]

        # subsampling to speed up
        np.random.seed(0)
        idx = np.arange(self.targets.size)
        np.random.shuffle(idx)
        idx = idx[:500]
        self.targets = self.targets[idx]
        self.data = self.data[idx]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                             random_state=0)
        # print(self.x_train[:10],self.y_train[:10])

        # normalize the training set
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)

    def support_vector_regression(self):
        # C = np.logspace(-3,3,num=7)
        # gamma = np.logspace(-3,3,num=7)
        #
        # svr = Support_vector_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     # n_iter=30,
        #     n_jobs=10,
        #     C= C,
        #     kernel=['sigmoid', 'rbf' , 'linear'],
        #     gamma= gamma,
        #     grid_search=True)
        # Parameter
        # range: {'C': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),
        #         'kernel': ['sigmoid', 'rbf', 'linear'],
        #         'gamma': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]), 'coef0': (0.0,),
        #         'epsilon': (0.1,)}
        # Best
        # estimator: SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
        #                kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        # SVR: 0.00290
        np.random.seed(0)
        C = scipy.stats.norm(100, 100).rvs(100)
        C = C[C > 0]
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            n_jobs=10,
            C=C,
            kernel=['linear'],
            # gamma= scipy.stats.reciprocal(0.01, 20),
            random_search=True)

        # svr = Support_vector_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,)

        # svr.print_parameter_candidates()
        # svr.print_best_estimator()

        return (svr.evaluate(data=self.x_train, targets=self.y_train),
                svr.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_regression(self):
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            # n_iter=50,
            max_depth=range(1, 20),
            min_samples_leaf=range(1, 20),
            n_jobs=10,
            grid_search=True)
        # dtr = Decision_tree_regressor(self.x_train,self.y_train)

        # dtr.print_parameter_candidates()
        # dtr.print_best_estimator()

        return (dtr.evaluate(data=self.x_train, targets=self.y_train),
                dtr.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_regression(self):

        # n_estimators = np.logspace(1,4,4,dtype=np.int)
        # max_depth = np.logspace(2,6,5,base=2,dtype=np.int)
        # rfr = Random_forest_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     n_jobs=10,
        #     n_estimators=n_estimators,
        #     max_depth=max_depth,
        #     grid_search=True)
        # Parameter
        # range: {'n_estimators': array([10, 100, 1000, 10000]), 'max_depth': array([4, 8, 16, 32, 64])}
        # Best
        # estimator: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=32,
        #                                  max_features='auto', max_leaf_nodes=None,
        #                                  min_impurity_decrease=0.0, min_impurity_split=None,
        #                                  min_samples_leaf=1, min_samples_split=2,
        #                                  min_weight_fraction_leaf=0.0, n_estimators=1000,
        #                                  n_jobs=None, oob_score=False, random_state=0, verbose=0,
        #                                  warm_start=False)
        # RFR: 0.99931, 21.46442
        np.random.seed(0)
        n_estimators = scipy.stats.norm(1000,100).rvs(10).astype(np.int)
        max_depth = scipy.stats.norm(32,10).rvs(10).astype(np.int)
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=10,
            n_iter= 10,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_search=True)

        # rfr = Random_forest_regressor(self.x_train,self.y_train)

        # rfr.print_parameter_candidates()
        # rfr.print_best_estimator()

        return (rfr.evaluate(data=self.x_train, targets=self.y_train),
                rfr.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_regression(self):

        # n_estimators = np.logspace(start=1, stop=4, base=10, num=4, dtype=np.int)
        # lr = np.logspace(-3, 3, num=7)
        # abr = Ada_boost_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     n_estimators=n_estimators,
        #     learning_rate=lr,
        #     n_jobs=10,
        #     grid_search=True)

        # Parameter
        # range: {'n_estimators': array([10, 100, 1000, 10000]),
        #         'learning_rate': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03])}
        # Best
        # estimator: AdaBoostRegressor(base_estimator=None, learning_rate=0.01, loss='linear',
        #                              n_estimators=10000, random_state=0)
        # ABR: 0.97866, 662.26056

        np.random.seed(0)
        n_estimators = scipy.stats.norm(10000,1000).rvs(10).astype(np.int)
        lr = scipy.stats.norm(0.01,0.01).rvs(100)
        lr = lr[lr>0]
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            n_estimators=n_estimators,
            learning_rate=lr,
            n_jobs=10,
            random_search= True)
        # abr = Ada_boost_regressor(self.x_train,self.y_train)


        # abr.print_parameter_candidates()
        # abr.print_best_estimator()

        return (abr.evaluate(data=self.x_train, targets=self.y_train),
                abr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_process_regression(self):
        # alpha = np.logspace(start=-2, stop=2, num=5, dtype=np.float32)
        # kernel = (1.0 * RBF(1.0), 1.0 * RBF(0.5), WhiteKernel())
        # gpr = Gaussian_process_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     n_iter=10,
        #     kernel=kernel,
        #     alpha=alpha,
        #     n_jobs=10,
        #     grid_search=True)

        # Parameter
        # range: {'kernel': (1 ** 2 * RBF(length_scale=1), 1 ** 2 * RBF(length_scale=0.5), WhiteKernel(noise_level=1)),
        #         'alpha': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02], dtype=float32)}
        # Best
        # estimator: GaussianProcessRegressor(alpha=0.01, copy_X_train=True,
        #                                     kernel=1 ** 2 * RBF(length_scale=0.5),
        #                                     n_restarts_optimizer=0, normalize_y=False,
        #                                     optimizer='fmin_l_bfgs_b', random_state=0)
        # GPR: 1.00000, 0.00191

        np.random.seed(0)
        alpha = scipy.stats.norm(0.01, 0.01).rvs(100)
        alpha = alpha[alpha > 0].round(5).astype(np.float32)

        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=5,
            kernel=(1.0 * RBF(0.5),),
            alpha=alpha,
            n_jobs=5,
            random_search= True)

        # print all possible parameter values and the best parameters
        # gpr.print_parameter_candidates()
        # gpr.print_best_estimator()



        # return the mean squared error
        return (gpr.evaluate(data=self.x_train, targets=self.y_train),
                gpr.evaluate(data=self.x_test, targets=self.y_test))

    def linear_regression(self):
        # alpha = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)
        # max_iter = np.logspace(start=2, stop=4, base=10, num=3, dtype=np.int)
        # solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')
        # lr = Linear_least_squares(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     n_jobs = 10,
        #     alpha = alpha,
        #     max_iter = max_iter,
        #     solver= solver,
        #     grid_search= True)

        # Parameter
        # range: {'alpha': array([1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03], dtype=float32),
        #         'max_iter': array([100, 1000, 10000]), 'solver': ('auto', 'svd', 'cholesky', 'lsqr', 'saga')}
        # Best
        # estimator: Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=100, normalize=False,
        #                  random_state=0, solver='auto', tol=0.001)
        # LR: 1.00000, 0.00237
        np.random.seed(0)
        alpha = scipy.stats.norm(0.1,0.1).rvs(100).astype(np.float32)
        alpha = alpha[alpha>0].round(5)
        max_iter = scipy.stats.norm(100,10).rvs(100).astype(np.int)
        max_iter = max_iter[max_iter>0]

        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            n_jobs = 10,
            n_iter= 10,
            alpha = alpha,
            max_iter = max_iter,
            solver= ["auto"],
            random_search= True)

        # lr.print_parameter_candidates()
        # lr.print_best_estimator()

        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_regression(self):
        np.random.seed(0)
        # hidden_layer_units = np.logspace(start=5, stop=7, base=2, num=3, dtype=np.int)
        # hidden_layer_sizes = hidden_layer_units
        # max_iter = np.logspace(start=3,stop=4, base=10, num=2, dtype=np.int)
        # nnr = Neural_network_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     n_iter=30,
        #     hidden_layer_sizes=hidden_layer_sizes,
        #     max_iter=max_iter,
        #     n_jobs=10,
        #     grid_search=True)

        # Parameter
        # range: {'hidden_layer_sizes': array([32, 64, 128]), 'activation': ('relu',), 'max_iter': array([1000, 10000]),
        #         'batch_size': ('auto',)}
        # Best
        # estimator: MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
        #                         beta_2=0.999, early_stopping=False, epsilon=1e-08,
        #                         hidden_layer_sizes=32, learning_rate='constant',
        #                         learning_rate_init=0.001, max_iter=10000, momentum=0.9,
        #                         n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
        #                         random_state=0, shuffle=True, solver='adam', tol=0.0001,
        #                         validation_fraction=0.1, verbose=False, warm_start=False)
        # NNR: 0.99873, 40.06854

        hidden_layer_sizes = scipy.stats.norm(32,10).rvs(100).astype(np.int)
        hidden_layer_sizes = hidden_layer_sizes[hidden_layer_sizes>0]
        max_iter = scipy.stats.norm(10000,1000).rvs(100).astype(np.int)
        max_iter = max_iter[max_iter>0]

        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            n_jobs=10,
            random_search=True)



        # print all possible parameter values and the best parameters
        # nnr.print_parameter_candidates()
        # nnr.print_best_estimator()

        # return the mean squared error
        return (nnr.evaluate(data=self.x_train, targets=self.y_train),
                nnr.evaluate(data=self.x_test, targets=self.y_test))

    def printself(self):
        print(self.data)
        print(self.targets)


if __name__ == '__main__':
    bs = Bike_Sharing()
    
    # retrieve the results
    svr_results = bs.support_vector_regression()
    dtr_results = bs.decision_tree_regression()
    rfr_results = bs.random_forest_regression()
    abr_results = bs.ada_boost_regression()
    gpr_results = bs.gaussian_process_regression()
    lls_results = bs.linear_regression()
    nnr_results = bs.neural_network_regression()

    print("(mean_square_error, r2_score) on training set:")
    print('SVR: (%.3f, %.3f)' % (svr_results[0]))
    print('DTR: (%.3f, %.3f)' % (dtr_results[0]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[0]))
    print('ABR: (%.3f, %.3f)' % (abr_results[0]))
    print('GPR: (%.3f, %.3f)' % (gpr_results[0]))
    print('LLS: (%.3f, %.3f)' % (lls_results[0]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[0]))

    print("(mean_square_error, r2_score) on test set:")
    print('SVR: (%.3f, %.3f)' % (svr_results[1]))
    print('DTR: (%.3f, %.3f)' % (dtr_results[1]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[1]))
    print('ABR: (%.3f, %.3f)' % (abr_results[1]))
    print('GPR: (%.3f, %.3f)' % (gpr_results[1]))
    print('LLS: (%.3f, %.3f)' % (lls_results[1]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[1]))
