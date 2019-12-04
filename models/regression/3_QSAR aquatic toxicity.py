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


class QSAR_aquatic_toxicity:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/3_QSAR_aquatic_toxicity'
        filename = 'qsar_aquatic_toxicity.csv'

        f = np.loadtxt(os.path.join(settings.ROOT_DIR, filepath, filename),
            delimiter=';')
        self.data = f[:,:8]
        self.targets = f[:,8:].reshape(-1)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

    def support_vector_regression(self):
        np.random.seed(0)
        kernel = ('sigmoid', 'rbf')
        C = scipy.stats.reciprocal(1, 100)
        gamma = scipy.stats.reciprocal(0.01, 20)
        coef0 = scipy.stats.uniform(0, 5)
        epsilon = scipy.stats.reciprocal(0.01, 1)

        svr = Support_vector_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            epsilon=epsilon,
            random_search=True)

        #svr.print_parameter_candidates()
        #svr.print_best_estimator()

        return (svr.evaluate(data=self.x_train, targets=self.y_train),
                svr.evaluate(data=self.x_test, targets=self.y_test))

    def decision_tree_regression(self):
        max_depth = range(1, 14)
        min_samples_leaf = range(1, 9)

        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            grid_search=True)

        #dtr.print_parameter_candidates()
        #dtr.print_best_estimator()

        return (dtr.evaluate(data=self.x_train, targets=self.y_train),
                dtr.evaluate(data=self.x_test, targets=self.y_test))

    def random_forest_regression(self):
        n_estimators = range(1, 20)
        max_depth = range(1, 20)

        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            grid_search=True)

        #rfr.print_parameter_candidates()
        #rfr.print_best_estimator()

        return (rfr.evaluate(data=self.x_train, targets=self.y_train),
                rfr.evaluate(data=self.x_test, targets=self.y_test))

    def ada_boost_regression(self):
        n_estimators = range(1,100)
        abr = Ada_boost_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=99,
            n_estimators=n_estimators,
            random_search=True)

        #abr.print_parameter_candidates()
        #abr.print_best_estimator()

        return (abr.evaluate(data=self.x_train, targets=self.y_train),
                abr.evaluate(data=self.x_test, targets=self.y_test))

    def gaussian_process_regression(self):
        gpr = Gaussian_process_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=50,
            alpha=scipy.stats.reciprocal(1e-11, 1e-8),
            n_jobs=10,
            random_search=True)

        # print all possible parameter values and the best parameters
        #gpr.print_parameter_candidates()
        #gpr.print_best_estimator()

        # return the mean squared error
        return (gpr.evaluate(data=self.x_train, targets=self.y_train),
                gpr.evaluate(data=self.x_test, targets=self.y_test))

    def linear_regression(self):
        lr = Linear_least_squares(
            x_train=self.x_train,
            y_train=self.y_train,
            alpha=scipy.stats.reciprocal(1,1000),
            cv=3,
            n_iter=99,
            random_search=True)
        #lr.print_parameter_candidates()
        #lr.print_best_estimator()

        return (lr.evaluate(data=self.x_train, targets=self.y_train),
                lr.evaluate(data=self.x_test, targets=self.y_test))

    def neural_network_regression(self):
        hidden_layer_sizes = []
        for i in range(3, 40):
            hidden_layer_sizes.append((i,))
        hidden_layer_sizes = np.asarray(hidden_layer_sizes)
        batch_size = range(5, 200)
        mlp=Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            activation=('tanh',),
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            cv=3,
            n_iter=100,
            n_jobs=10,
            random_search=True
        )
        #mlp.print_parameter_candidates()
        #mlp.print_best_estimator()

        return (mlp.evaluate(data=self.x_train, targets=self.y_train),
                mlp.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    qsar = QSAR_aquatic_toxicity()

    # retrieve the results
    svr_results = qsar.support_vector_regression()
    dtr_results = qsar.decision_tree_regression()
    rfr_results = qsar.random_forest_regression()
    abr_results = qsar.ada_boost_regression()
    gpr_results = qsar.gaussian_process_regression()
    lls_results = qsar.linear_regression()
    nnr_results = qsar.neural_network_regression()

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

