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

        #dtr.print_parameter_candidates()
        #dtr.print_best_estimator()

        return (dtr.evaluate(data=self.X_train, targets=self.y_train),
                dtr.evaluate(data=self.X_test, targets=self.y_test))

    def random_forest_regression(self):
        '''
        random forest take too long too train,here 50 is th best n estimator from the search
        '''
        rfr = Random_forest_regressor(
            x_train=self.X_train,
            y_train=self.y_train,
            n_estimators=50,
            max_depth=8)

        return (rfr.evaluate(data=self.X_train, targets=self.y_train),
                rfr.evaluate(data=self.X_test, targets=self.y_test))

    def ada_boost_regression(self):
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(0, 4) :
            abr = Ada_boost_regressor(
                x_train=self.X_train,
                y_train=self.y_train_list[i],
                n_estimators=(50,),
                grid_search=True)

            #abr.print_parameter_candidates()
            #abr.print_best_estimator()
            # training
            train_r2.append('%.3f' % abr.r2_score(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            train_mse.append('%.3f' % abr.mean_squared_error(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            # test
            test_r2.append('%.3f' % abr.r2_score(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
            test_mse.append('%.3f' % abr.mean_squared_error(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        return res

    def gaussian_process_regression(self):
        np.random.seed(0)
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(0, 4):
            gpr = Gaussian_process_regressor(
                x_train=self.X_train,
                y_train=self.y_train,
                cv=3,
                n_iter=50,
                alpha=scipy.stats.reciprocal(1e-11, 1e-8),
                n_jobs=10,
                random_search=True)
            train_r2.append('%.3f' % gpr.r2_score(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            train_mse.append('%.3f' % gpr.mean_squared_error(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            # test
            test_r2.append('%.3f' % gpr.r2_score(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
            test_mse.append('%.3f' % gpr.mean_squared_error(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        # print all possible parameter values and the best parameters
        #gpr.print_parameter_candidates()
        #gpr.print_best_estimator()

        return res

    def linear_regression(self):
        np.random.seed(0)
        res=[]
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(0, 4):
            lr = Linear_least_squares(
                x_train=self.X_train,
                y_train=self.y_train_list[i],
                alpha=scipy.stats.reciprocal(1,1000),
                cv=3,
                n_iter=99,
                random_search=True)

            #lr.print_parameter_candidates()
            #lr.print_best_estimator()
            #training
            train_r2.append('%.3f'% lr.r2_score(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            train_mse.append('%.3f'% lr.mean_squared_error(
                x_test=self.X_train,
                y_test=self.y_train_list[i]))
            #test
            test_r2.append('%.3f'% lr.r2_score(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
            test_mse.append('%.3f'% lr.mean_squared_error(
                x_test=self.X_test,
                y_test=self.y_test_list[i]))
        res.append((train_mse,train_r2))
        res.append((test_mse, test_r2))
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
        #mlp.print_parameter_candidates()
        #mlp.print_best_estimator()

        return (mlp.evaluate(data=self.X_train, targets=self.y_train),
                mlp.evaluate(data=self.X_test, targets=self.y_test))



if __name__ == '__main__':
    sgemm = SGEMM()
    # retrieve the results
    dtr_results = sgemm.decision_tree_regression()
    rfr_results = sgemm.random_forest_regression()
    abr_results = sgemm.ada_boost_regression()
    gpr_results = sgemm.gaussian_process_regression()
    lls_results = sgemm.linear_regression()
    nnr_results = sgemm.neural_network_regression()

    print("(mean_square_error, r2_score) on training set:")
    print('DTR: (%.3f, %.3f)' % (dtr_results[0]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[0]))
    print('ABR:'+str(abr_results[0]))
    print('GPR:' +str(gpr_results[0]))
    print('LLS:'+str(lls_results[0]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[0]))

    print("(mean_square_error, r2_score) on test set:")
    print('DTR: (%.3f, %.3f)' % (dtr_results[1]))
    print('RFR: (%.3f, %.3f)' % (rfr_results[1]))
    print('ABR:'+str(abr_results[1]))
    print('GPR: ' +str(gpr_results[1]))
    print('LLS:' + str(lls_results[1]))
    print('NNR: (%.3f, %.3f)' % (nnr_results[1]))
