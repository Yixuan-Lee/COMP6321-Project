from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error


class Gaussian_process_regressor(Cross_validation):
    __gpr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10, n_jobs=None,
            kernel=(1.0*RBF(1.0),), alpha=(1e-10,),
            grid_search=False, random_search=False):

        self.__gpr = GaussianProcessRegressor(random_state=0)

        try:
            self.__param = {
                'kernel': kernel,
                'alpha': alpha
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__gpr = super().grid_search_cv(self.__gpr,
                        self.__param, cv, n_jobs, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__gpr = super().random_search_cv(self.__gpr,
                        self.__param, cv, n_iter, n_jobs, x_train, y_train)
                else:
                    # fit data directly
                    self.__gpr.fit(x_train, y_train)
        except:
            print("Gaussian_process_regressor: x_train or y_train may be wrong")

    def mean_squared_error(self, x_test=None, y_test=None):
        """
        get regression mean squared error

        :param x_test: test data
        :param y_test: test targets
        :return: the accuracy score
        """
        try:
            return mean_squared_error(
                y_true=y_test,
                y_pred=self.__gpr.predict(x_test))
        except:
            print("Gaussian_process_regressor: x_test or y_test may be wrong")

    def print_parameter_candidates(self):
        """
        print all possible parameter combinations
        """
        print('Parameter range: ', self.__param)

    def print_best_estimator(self):
        """
        print the best hyper-parameters
        """
        try:
            print('Best estimator : ', self.__gpr.best_estimator_)
        except:
            print("Gaussian_process_regressor: __gpr didn't use GridSearchCV "
                  "or RandomSearchCV.")
