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


class Merck_molecular:
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
#         C = np.logspace(-3, 3, num=7)
#         gamma = np.logspace(-3, 3, num=7)
#         svr = Support_vector_regressor(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             cv=3,
#             # n_iter=30,
#             n_jobs=-1,
#             C=C,
#             kernel=['sigmoid', 'rbf', 'linear'],
#             gamma=gamma,
#             scoring=self.scoring,
#             grid_search=True)
# ACT2
#         Best estimator :  SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
#     kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# ACT4
# Best estimator :  SVR(C=10.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
#     kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        np.random.seed(0)
        if self.data_option == "ACT2":
            C = scipy.stats.norm(0.1, 0.1).rvs(100)
            C = C[C > 0]
            gamma = scipy.stats.norm(0.001, 0.001).rvs(100)
            gamma = gamma[ gamma>0]
            kernel = ('sigmoid',)
        else:
            C = scipy.stats.norm(10, 1).rvs(100)
            C = C[C > 0]
            gamma = scipy.stats.norm(0.001, 0.001).rvs(100)
            gamma = gamma[ gamma>0]
            kernel = ('rbf',)
        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            n_jobs=-1,
            C=C,
            kernel=kernel,
            gamma= gamma,
            scoring=self.scoring,
            random_search=True)


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
            max_depth=range(1, 20 , 4),
            min_samples_leaf=range(1, 20 , 4),
            n_jobs=-1,
            scoring=self.scoring,
            grid_search=True)
#ATC2
# Best estimator :  DecisionTreeRegressor(criterion='mse', max_depth=6, max_features=None,
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=10,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       presort=False, random_state=0, splitter='best')
#ATC4
# Best estimator :  DecisionTreeRegressor(criterion='mse', max_depth=6, max_features=None,
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       presort=False, random_state=0, splitter='best')
        # dtr = Decision_tree_regressor(self.x_train,self.y_train)


        # dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return (self.score_func(self.y_train, dtr.predict(self.x_train)),
                self.score_func(self.y_test, dtr.predict(self.x_test)))

    def random_forest_regression(self):
#         n_estimators = np.logspace(3, 6, 4, base=2, dtype=np.int)
#         max_depth = np.logspace(3, 6, 4, base=2, dtype=np.int)
#         rfr = Random_forest_regressor(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             cv=3,
#             n_jobs=-1,
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             scoring=self.scoring,
#             grid_search=True)
#ACT2
# Best estimator :  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=16,
#                       max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=64,
#                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
#                       warm_start=False)
#ACT4
# Best estimator :  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=32,
#                       max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=64,
#                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
#                       warm_start=False)

        np.random.seed(0)
        if self.data_option == "ACT2":
            n_estimators = scipy.stats.norm(64, 100).rvs(100).astype(np.int)
            n_estimators = n_estimators[n_estimators>0]
            max_depth = scipy.stats.norm(16, 10).rvs(100).astype(np.int)
            max_depth = max_depth[max_depth>0]
        else:
            n_estimators = scipy.stats.norm(64, 100).rvs(100).astype(np.int)
            n_estimators = n_estimators[n_estimators>0]
            max_depth = scipy.stats.norm(32, 10).rvs(100).astype(np.int)
            max_depth = max_depth[max_depth>0]


        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            scoring=self.scoring,
            random_search=True)

        # rfr = Random_forest_regressor(self.x_train,self.y_train)

        # rfr.print_parameter_candidates()
        rfr.print_best_estimator()

        return (self.score_func(self.y_train, rfr.predict(self.x_train)),
                self.score_func(self.y_test, rfr.predict(self.x_test)))

    def ada_boost_regression(self):
#         n_estimators = np.logspace(3, 6, 4, base=2, dtype=np.int)
#         lr = np.logspace(-3, -1, num=3)
#         abr = Ada_boost_regressor(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             cv=3,
#             n_estimators=n_estimators,
#             learning_rate=lr,
#             n_jobs=-1,
#             scoring=self.scoring,
#             grid_search=True)

# ACT2
# Best estimator :  AdaBoostRegressor(base_estimator=None, learning_rate=0.1, loss='linear',
#                   n_estimators=64, random_state=0)
# ACT4
# Best estimator :  AdaBoostRegressor(base_estimator=None, learning_rate=0.01, loss='linear',
#                   n_estimators=64, random_state=0)


        np.random.seed(0)
        if self.data_option == "ACT2":
            n_estimators = scipy.stats.norm(64, 100).rvs(100).astype(np.int)
            n_estimators = n_estimators[n_estimators>0]
            lr = scipy.stats.norm(0.1, 0.01).rvs(100)
            lr = lr[lr > 0]
        else:
            n_estimators = scipy.stats.norm(64, 100).rvs(100).astype(np.int)
            n_estimators = n_estimators[n_estimators>0]
            lr = scipy.stats.norm(0.01, 0.001).rvs(100)
            lr = lr[lr > 0]

        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            n_estimators=n_estimators,
            learning_rate=lr,
            n_jobs=-1,
            scoring=self.scoring,
            random_search=True)
        # abr = Ada_boost_regressor(self.x_train,self.y_train)

        # abr.print_parameter_candidates()
        abr.print_best_estimator()

        return (self.score_func(self.y_train, abr.predict(self.x_train)),
                self.score_func(self.y_test, abr.predict(self.x_test)))

    def gaussian_process_regression(self):      
        
#         kernel = (sklearn.gaussian_process.kernels.RBF(10.0),
#                   sklearn.gaussian_process.kernels.RBF(5.0),
#                   sklearn.gaussian_process.kernels.RBF(1.0),
#                  )

#         alpha = np.logspace(start=-4, stop=0, num=5, dtype=np.float32)
#         gpr = Gaussian_process_regressor(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             cv=3,
#             #             n_iter=10,
#             kernel=kernel,
#             alpha=alpha,
#             n_jobs=-1,
#             scoring=self.scoring,
#             grid_search=True)

# Best estimator :  GaussianProcessRegressor(alpha=1.0, copy_X_train=True,
#                          kernel=RBF(length_scale=5), n_restarts_optimizer=0,
#                          normalize_y=True, optimizer='fmin_l_bfgs_b',
#                          random_state=0)
# ----------ACT2 RESULT----------
# gpr | training set : 0.727, | test set : 0.436.
# Best estimator :  GaussianProcessRegressor(alpha=0.1, copy_X_train=True,
#                          kernel=RBF(length_scale=10), n_restarts_optimizer=0,
#                          normalize_y=True, optimizer='fmin_l_bfgs_b',
#                          random_state=0)
# ----------ACT4 RESULT----------
# gpr | training set : 0.856, | test set : 0.552.

        np.random.seed(0)
        if self.data_option == "ACT2" :
            alpha = scipy.stats.norm(1, 0.1).rvs(100)
            alpha = alpha[alpha > 0].round(5).astype(np.float32)
            kernel = (sklearn.gaussian_process.kernels.RBF(5.0),)
        else : 
            alpha = scipy.stats.norm(0.1, 0.01).rvs(100)
            alpha = alpha[alpha > 0].round(5).astype(np.float32)
            kernel = (sklearn.gaussian_process.kernels.RBF(10.0),)
        
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter= 10,
            kernel= kernel,
            alpha=alpha,
            n_jobs=-1,
            scoring=self.scoring,
            random_search=True)

        # print all possible parameter values and the best parameters
        # gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return (self.score_func(self.y_train, gpr.predict(self.x_train)),
                self.score_func(self.y_test, gpr.predict(self.x_test)))

    def linear_regression(self):
#         alpha = np.logspace(start=-1, stop=3, base=10, num=5, dtype=np.float32)
#         max_iter = np.logspace(start=2, stop=4, base=10, num=3, dtype=np.int)
#         solver = ('auto', 'svd', 'cholesky', 'lsqr', 'saga')
#         lr = Linear_least_squares(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             n_jobs=-1,
#             alpha=alpha,
#             max_iter=max_iter,
#             solver=solver,
#             scoring=self.scoring,
#             grid_search=True)

# ACT2
# Best estimator :  Ridge(alpha=1000.0, copy_X=True, fit_intercept=True, max_iter=100,
#       normalize=False, random_state=0, solver='lsqr', tol=0.001)
# ACT4
# Best estimator :  Ridge(alpha=1000.0, copy_X=True, fit_intercept=True, max_iter=100,
#       normalize=False, random_state=0, solver='lsqr', tol=0.001)

        np.random.seed(0)
        alpha = scipy.stats.norm(1000, 100).rvs(100).astype(np.float32)
        alpha = alpha[alpha > 0].round(5)
        max_iter = scipy.stats.norm(100, 10).rvs(100).astype(np.int)


        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            n_jobs=-1,
            n_iter=10,
            alpha=alpha,
            max_iter=max_iter,
            solver=["lsqr"],
            scoring=self.scoring,
            random_search=True)

        # lr.print_parameter_candidates()
        lr.print_best_estimator()

        return (self.score_func(self.y_train, lr.predict(self.x_train)),
                self.score_func(self.y_test, lr.predict(self.x_test)))

    def neural_network_regression(self):

#         hidden_layer_units = np.logspace(start=5, stop=7, base=2, num=3, dtype=np.int)
#         hidden_layer_sizes = hidden_layer_units
#         max_iter = np.logspace(start=3, stop=4, base=10, num=2, dtype=np.int)
#         nnr = Neural_network_regressor(
#             x_train=self.x_train,
#             y_train=self.y_train,
#             cv=3,
#             n_iter=30,
#             hidden_layer_sizes=hidden_layer_sizes,
#             max_iter=max_iter,
#             n_jobs=-1,
#             scoring=self.scoring,
#             grid_search=True)
# ACT2
# Best estimator :  MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#              beta_2=0.999, early_stopping=False, epsilon=1e-08,
#              hidden_layer_sizes=32, learning_rate='constant',
#              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
#              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#              random_state=0, shuffle=True, solver='adam', tol=0.0001,
#              validation_fraction=0.1, verbose=False, warm_start=False)
# ACT4
# Best estimator :  MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#              beta_2=0.999, early_stopping=False, epsilon=1e-08,
#              hidden_layer_sizes=64, learning_rate='constant',
#              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
#              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#              random_state=0, shuffle=True, solver='adam', tol=0.0001,
#              validation_fraction=0.1, verbose=False, warm_start=False)
        np.random.seed(0)
        if self.data_option == "ACT2" :
            hidden_layer_sizes = scipy.stats.norm(32, 10).rvs(100).astype(np.int)
            hidden_layer_sizes = hidden_layer_sizes[hidden_layer_sizes > 0]
            max_iter = scipy.stats.norm(1000, 100).rvs(100).astype(np.int)
            max_iter = max_iter[max_iter > 0]
        else :
            hidden_layer_sizes = scipy.stats.norm(64, 10).rvs(100).astype(np.int)
            hidden_layer_sizes = hidden_layer_sizes[hidden_layer_sizes > 0]
            max_iter = scipy.stats.norm(1000, 100).rvs(100).astype(np.int)
            max_iter = max_iter[max_iter > 0]

        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=30,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            n_jobs=-1,
            scoring=self.scoring,
            random_search=True)

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
    def add_dic(dic,**kwargs):
        dic.update(kwargs)

    mm1 = Merck_molecular(1)
    mm2 = Merck_molecular(2)
    mm_list = []
    mm_list.append(mm1)
    mm_list.append(mm2)

    rs_final = []


    for mm in mm_list:
        rs = {}
        svr_results = call_with_timeout(180,mm.support_vector_regression)
        dtr_results = call_with_timeout(180,mm.decision_tree_regression)
        rfr_results = call_with_timeout(180,mm.random_forest_regression)
        abr_results = call_with_timeout(180,mm.ada_boost_regression)
        gpr_results = call_with_timeout(180,mm.gaussian_process_regression)
        llr_results = call_with_timeout(180,mm.linear_regression)
        nnr_results = call_with_timeout(180,mm.neural_network_regression)
        add_dic(rs,
                svr = svr_results,
                dtr = dtr_results,
                rfr = rfr_results,
               abr = abr_results,
               gpr = gpr_results,
               llr = llr_results,
               nnr = nnr_results
               )
        rs_final.append(rs)
    #     print("----------%s RESULT----------" % mm.data_option)
    #     for key,value in rs.items():
    #         if None in value:
    #             print("Error: %s regressor timed out after 3 minutes" % key)
    #         else:
    #             print('%s | training set : %.3f, | test set : %.3f.' % (key,value[0],value[1]))
    print("----------RESULT----------")
    for key,value in rs_final[0].items():
        rs1 = np.array(rs_final[0].get(key))
        rs2 = np.array(rs_final[1].get(key))
        if None in rs1 or None in rs2:
            print("Error: %s regressor timed out after 3 minutes" % key)
        else:
            rs = (rs1 + rs2)/2
            print('%s | training set : %.3f, | test set : %.3f.' % (key,rs[0],rs[1]))
