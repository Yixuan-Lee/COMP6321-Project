import os
import scipy
import numpy as np
import scipy.stats              # For reciprocal distribution
from models import settings     # For retrieving root path
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_least_squares import Linear_least_squares
from neural_network_regressor import Neural_network_regressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SGEMM :
    data = []
    targets = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    y_train_list = []
    y_test_list = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/9_SGEMM_GPU_kernel_performance'
        filename = 'sgemm_product.csv'

        self.data = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename),
            delimiter=',',skiprows=1)
        X = self.data[:, :14]
        target = self.data[:, 14:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, target, test_size=0.33, random_state=0)
        scaler = StandardScaler()
        X_train_preffix = scaler.fit_transform(self.X_train[:, :-4])
        X_test_preffix = scaler.transform(self.X_test[:, :-4])
        self.X_train[:, :-4] = X_train_preffix
        self.X_test[:, :-4] = X_test_preffix


        self.y_train_list.append(self.y_train[:, :1].ravel())
        self.y_train_list.append(self.y_train[:, 1:2].ravel())
        self.y_train_list.append(self.y_train[:, 2:3].ravel())
        self.y_train_list.append(self.y_train[:, 3:4].ravel())

        self.y_test_list.append(self.y_test[:, :1].ravel())
        self.y_test_list.append(self.y_test[:, 1:2].ravel())
        self.y_test_list.append(self.y_test[:, 2:3].ravel())
        self.y_test_list.append(self.y_test[:, 3:4].ravel())
        self.y_train_list = np.asarray(self.y_train_list)
        self.y_test_list = np.asarray(self.y_test_list)



    def decision_tree_regression(self):
        max_depth = range(9, 20)
        min_samples_leaf = range(1, 9)

        dtr = Decision_tree_regressor(
            x_train=self.X_train,
            y_train=self.y_train,
            cv=3,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_iter=50,
            random_search=True)

        dtr.print_parameter_candidates()
        dtr.print_best_estimator()

        return dtr.r2_score(
            x_test=self.X_test,
            y_test=self.y_test)

    def random_forest_regression(self):
        rfr = Random_forest_regressor(
            x_train=self.X_train,
            y_train=self.y_train,
            n_estimators=50,
            max_depth=8)

        return rfr.r2_score(
            x_test=self.X_test,
            y_test=self.y_test)

    def ada_boost_regression(self):
        res=[]
        for i in range(0, 4) :
            abr = Ada_boost_regressor(
                x_train=self.X_train,
                y_train=self.y_train_list[i],
                n_estimators=50,
                random_search=True)

            abr.print_parameter_candidates()
            abr.print_best_estimator()
            res.append(abr.r2_score(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
        return res

    def gaussian_process_regression(self):
        gpr = Gaussian_process_regressor(
            x_train=self.X_train,
            y_train=self.y_train,
            cv=3,
            n_iter=50,
            alpha=scipy.stats.reciprocal(1e-11, 1e-8),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        gpr.print_parameter_candidates()
        gpr.print_best_estimator()

        # return the mean squared error
        return gpr.r2_score(
            x_test=self.X_test,
            y_test=self.y_test)

    def linear_regression(self):
        np.random.seed(0)
        res = []
        for i in range(0, 4):
            lr = Linear_least_squares(
                x_train=self.X_train,
                y_train=self.y_train_list[i],
                alpha=scipy.stats.reciprocal(1,1000),
                cv=3,
                n_iter=99,
                random_search=True)

            lr.print_parameter_candidates()
            lr.print_best_estimator()
            res.append(lr.r2_score(
                x_test=self.X_test,
                y_test=self.y_test_[i]))
        return res

    def neural_network_regression(self):
        mlp=Neural_network_regressor(
            x_train=self.X_train,
            y_train=self.y_train,
            activation='tanh',
            hidden_layer_sizes=(14,),
            batch_size=range(5,200),
            cv=3,
            n_iter=100,
            n_jobs=10,
            random_search=True
        )
        mlp.print_parameter_candidates()
        mlp.print_best_estimator()

        return mlp.r2_score(
            x_test=self.X_test,
            y_test=self.y_test)



if __name__ == '__main__':
    sgemm = SGEMM()
    print("mean squared error on the actual test set:")
    print('DTR: %.5f' % (sgemm.decision_tree_regression()))
    print('RFR: %.5f' % (sgemm.random_forest_regression()))
    print('ABR: %.5f' % (sgemm.ada_boost_regression()))
    print('GPR: %.5f' % (sgemm.gaussian_process_regression()))
    print(' LR: %.5f' % (sgemm.linear_regression()))
    print('NNR: %.5f' % (sgemm.neural_network_regression()))