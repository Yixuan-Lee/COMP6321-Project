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


class Merck_Molecular:
    data_option = ""
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    scoring = None

    def __init__(self, option=1):

        '''
        :param option: if set option to 1, by using ACT2
                       if set option to 2, by using ACT4
        '''

        np.set_printoptions(threshold=np.inf)
        filepath = 'datasets/regression_datasets/10_Merck_Molecular_Activity_Challenge'

        if option == 1:
            MERCK_FILE = 'Merck_Data1.npz'
            self.data_option = "ACT2"
        else:
            MERCK_FILE = 'Merck_Data2.npz'
            self.data_option = "ACT4"
        MERCK_FILE = np.load(os.path.join(settings.ROOT_DIR, filepath, MERCK_FILE))
        X = MERCK_FILE.get("X")
        y = MERCK_FILE.get("y")
        np.random.seed(0)
        idx = np.arange(y.size)
        np.random.shuffle(idx)
        idx = idx[:500]
        X = X[idx]
        y = y[idx]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33,
                             random_state=0)
        # print(self.x_train[:10],self.y_train[:10])

        # normalize the training set
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = scaler.transform(self.x_test)
        self.scoring = sklearn.metrics.make_scorer(self.score_func)

    def support_vector_regression(self):
        C = np.logspace(-3, 3, num=7)
        gamma = np.logspace(-3, 3, num=7)
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            # n_iter=30,
            n_jobs=-1,
            C=C,
            kernel=['sigmoid', 'rbf', 'linear'],
            gamma=gamma,
            scoring=self.scoring,
            grid_search=True)
        # ACT2
        # Parameter
        # range: {'C': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),
        #         'kernel': ['sigmoid', 'rbf', 'linear'],
        #         'gamma': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]), 'coef0': (0.0,),
        #         'epsilon': (0.1,)}
        # Best
        # estimator: SVR(C=0.001, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
        #                kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        # np.random.seed(0)
        # C = scipy.stats.norm(100, 100).rvs(100)
        # C = C[C > 0]
        # svr = Support_vector_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,
        #     n_iter=30,
        #     n_jobs=10,
        #     C=C,
        #     kernel=['linear'],
        #     # gamma= scipy.stats.reciprocal(0.01, 20),
        #     random_search=True)

        # svr = Support_vector_regressor(
        #     x_train=self.x_train,
        #     y_train=self.y_train,
        #     cv=3,)

        # svr.print_parameter_candidates()
        svr.print_best_estimator()

        return (self.score_func(self.y_train, svr.predict(self.x_train)),
                self.score_func(self.y_test, svr.predict(self.x_test)))

    def decision_tree_regression(self):
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            # n_iter=50,
            max_depth=range(1, 20),
            min_samples_leaf=range(1, 20),
            n_jobs=-1,
            scoring=self.scoring,
            grid_search=True)
        # dtr = Decision_tree_regressor(self.x_train,self.y_train)

        # dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return (self.score_func(self.y_train, dtr.predict(self.x_train)),
                self.score_func(self.y_test, dtr.predict(self.x_test)))

    def random_forest_regression(self):
        n_estimators = np.logspace(3, 6, 4, base=2, dtype=np.int)
        max_depth = np.logspace(3, 6, 4, base=2, dtype=np.int)
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            scoring=self.scoring,
            grid_search=True)
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
        #         np.random.seed(0)
        #         n_estimators = scipy.stats.norm(1000, 100).rvs(10).astype(np.int)
        #         max_depth = scipy.stats.norm(32, 10).rvs(10).astype(np.int)
        #         rfr = Random_forest_regressor(
        #             x_train=self.x_train,
        #             y_train=self.y_train,
        #             cv=3,
        #             n_jobs=10,
        #             n_iter=10,
        #             n_estimators=n_estimators,
        #             max_depth=max_depth,
        #             random_search=True)

        # rfr = Random_forest_regressor(self.x_train,self.y_train)

        # rfr.print_parameter_candidates()
        rfr.print_best_estimator()

        return (self.score_func(self.y_train, rfr.predict(self.x_train)),
                self.score_func(self.y_test, rfr.predict(self.x_test)))

    def ada_boost_regression(self):
        n_estimators = np.logspace(3, 6, 4, base=2, dtype=np.int)
        lr = np.logspace(-3, -1, num=3)
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_estimators=n_estimators,
            learning_rate=lr,
            n_jobs=-1,
            scoring=self.scoring,
            grid_search=True)

        # Parameter
        # range: {'n_estimators': array([10, 100, 1000, 10000]),
        #         'learning_rate': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03])}
        # Best
        # estimator: AdaBoostRegressor(base_estimator=None, learning_rate=0.01, loss='linear',
        #                              n_estimators=10000, random_state=0)
        # ABR: 0.97866, 662.26056

        #         np.random.seed(0)
        #         n_estimators = scipy.stats.norm(10000, 1000).rvs(10).astype(np.int)
        #         lr = scipy.stats.norm(0.01, 0.01).rvs(100)
        #         lr = lr[lr > 0]
        #         abr = Ada_boost_regressor(
        #             x_train=self.x_train,
        #             y_train=self.y_train,
        #             cv=3,
        #             n_iter=10,
        #             n_estimators=n_estimators,
        #             learning_rate=lr,
        #             n_jobs=10,
        #             random_search=True)
        # abr = Ada_boost_regressor(self.x_train,self.y_train)

        # abr.print_parameter_candidates()
        abr.print_best_estimator()

        return (self.score_func(self.y_train, abr.predict(self.x_train)),
                self.score_func(self.y_test, abr.predict(self.x_test)))

    def gaussian_process_regression(self):
        alpha = np.logspace(start=-3, stop=0, num=4, dtype=np.float32)
        kernel = (1.0 * RBF(1.0), 1.0 * RBF(0.5), WhiteKernel())
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            #             n_iter=10,
            kernel=kernel,
            alpha=alpha,
            n_jobs=-1,
            scoring=self.scoring,
            grid_search=True)

        # Parameter
        # range: {'kernel': (1 ** 2 * RBF(length_scale=1), 1 ** 2 * RBF(length_scale=0.5), WhiteKernel(noise_level=1)),
        #         'alpha': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02], dtype=float32)}
        # Best
        # estimator: GaussianProcessRegressor(alpha=0.01, copy_X_train=True,
        #                                     kernel=1 ** 2 * RBF(length_scale=0.5),
        #                                     n_restarts_optimizer=0, normalize_y=False,
        #                                     optimizer='fmin_l_bfgs_b', random_state=0)
        # GPR: 1.00000, 0.00191

        #         np.random.seed(0)
        #         alpha = scipy.stats.norm(0.01, 0.01).rvs(100)
        #         alpha = alpha[alpha > 0].round(5).astype(np.float32)

        #         gpr = Gaussian_process_regressor(
        #             x_train=self.x_train,
        #             y_train=self.y_train,
        #             cv=3,
        #             n_iter=5,
        #             kernel=(1.0 * RBF(0.5),),
        #             alpha=alpha,
        #             n_jobs=5,
        #             random_search=True)

        # print all possible parameter values and the best parameters
        # gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return (self.score_func(self.y_train, gpr.predict(self.x_train)),
                self.score_func(self.y_test, gpr.predict(self.x_test)))

    def linear_regression(self):
        alpha = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)
        max_iter = np.logspace(start=2, stop=4, base=10, num=3, dtype=np.int)
        solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')
        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            n_jobs=-1,
            alpha=alpha,
            max_iter=max_iter,
            solver=solver,
            scoring=self.scoring,
            grid_search=True)

        # Parameter
        # range: {'alpha': array([1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03], dtype=float32),
        #         'max_iter': array([100, 1000, 10000]), 'solver': ('auto', 'svd', 'cholesky', 'lsqr', 'saga')}
        # Best
        # estimator: Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=100, normalize=False,
        #                  random_state=0, solver='auto', tol=0.001)
        # LR: 1.00000, 0.00237
        #         np.random.seed(0)
        #         alpha = scipy.stats.norm(0.1, 0.1).rvs(100).astype(np.float32)
        #         alpha = alpha[alpha > 0].round(5)
        #         max_iter = scipy.stats.norm(100, 10).rvs(100).astype(np.int)
        #         max_iter = max_iter[max_iter > 0]

        #         lr = Linear_least_squares(
        #             x_train=self.x_train,
        #             y_train=self.y_train,
        #             n_jobs=10,
        #             n_iter=10,
        #             alpha=alpha,
        #             max_iter=max_iter,
        #             solver=["auto"],
        #             random_search=True)

        # lr.print_parameter_candidates()
        lr.print_best_estimator()

        return (self.score_func(self.y_train, lr.predict(self.x_train)),
                self.score_func(self.y_test, lr.predict(self.x_test)))

    def neural_network_regression(self):
        np.random.seed(0)
        hidden_layer_units = np.logspace(start=5, stop=7, base=2, num=3, dtype=np.int)
        hidden_layer_sizes = hidden_layer_units
        max_iter = np.logspace(start=3, stop=4, base=10, num=2, dtype=np.int)
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            n_jobs=-1,
            scoring=self.scoring,
            grid_search=True)

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

        #         hidden_layer_sizes = scipy.stats.norm(32, 10).rvs(100).astype(np.int)
        #         hidden_layer_sizes = hidden_layer_sizes[hidden_layer_sizes > 0]
        #         max_iter = scipy.stats.norm(10000, 1000).rvs(100).astype(np.int)
        #         max_iter = max_iter[max_iter > 0]

        #         nnr = Neural_network_regressor(
        #             x_train=self.x_train,
        #             y_train=self.y_train,
        #             cv=3,
        #             n_iter=10,
        #             hidden_layer_sizes=hidden_layer_sizes,
        #             max_iter=max_iter,
        #             n_jobs=10,
        #             random_search=True)

        # print all possible parameter values and the best parameters
        # nnr.print_parameter_candidates()
        nnr.print_best_estimator()

        # return the mean squared error
        return (self.score_func(self.y_train, nnr.predict(self.x_train)),
                self.score_func(self.y_test, nnr.predict(self.x_test)))

    def printself(self):
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)

    @staticmethod
    def score_func(y, y_pred, **kwargs):
        rs = scipy.stats.pearsonr(y, y_pred)[0]
        rs = rs ** 2
        return rs


if __name__ == '__main__':
    mm = Merck_Molecular(1)
    # mm.printself()
    # mm2 = Merck_Molecular(2)
    # mm2.printself()

    # retrieve the results
    svr_results_ACT2 = mm.support_vector_regression()

    # svr_results_ACT4 = mm2.support_vector_regression()

    # dtr_results = mm.decision_tree_regression()
    # rfr_results = mm.random_forest_regression()
    # abr_results = mm.ada_boost_regression()
    # gpr_results = mm.gaussian_process_regression()
    # lls_results = mm.linear_least_squares()
    # nnr_results = mm.neural_network_regression()
    print("-----ACT2 RESULT-----")
    print("(correlation coefficient R2) on training set:")
    print('SVR: (%.3f)' % (svr_results_ACT2[0]))
    print("(correlation coefficient R2) on test set:")
    print('SVR: (%.3f)' % (svr_results_ACT2[1]))

    # print("-----ACT4 RESULT-----")
    # print("(mean_square_error, r2_score) on training set:")
    # print('SVR: (%.3f, %.3f)' % (svr_results_ACT4[0]))
    # print("(mean_square_error, r2_score) on test set:")
    # print('SVR: (%.3f, %.3f)' % (svr_results_ACT4[1]))
