from sklearn.linear_model import Ridge
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error, r2_score


class Linear_least_squares(Cross_validation):
    __lls = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10, n_jobs=None,
            alpha=(1.0,), max_iter=(None,), solver=('auto',),
            grid_search=False, random_search=False):

        self.__lls = Ridge(random_state=0)

        try:
            self.__param = {
                'alpha': alpha,
                'max_iter': max_iter,
                'solver': solver
            }

            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__lls = super().grid_search_cv(self.__lls,
                        self.__param, cv, n_jobs, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__lls = super().random_search_cv(self.__lls,
                        self.__param, cv, n_iter, n_jobs, x_train, y_train)
                else:
                    # fit data directly
                    self.__lls.fit(x_train, y_train)
        except:
            print("Linear_least_squares: x_train or y_train may be wrong")

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
                y_pred=self.__lls.predict(x_test))
        except:
            print("Linear_least_squares: x_test or y_test may be wrong")

    def r2_score(self, x_test=None, y_test=None):
        """
        get regression r2 score

        :param x_test: test data
        :param y_test: test targets
        :return: the r2 score
        """
        try:
            return r2_score(
                y_true=y_test,
                y_pred=self.__lls.predict(x_test))
        except:
            print("Linear_least_squares: x_test or y_test may be wrong")

    def evaluate(self, data=None, targets=None):
        """
        evaluate the model

        :param data: training or testing data
        :param targets: targets

        :return: return (mean_square_error, r2_score)
        """
        return (self.mean_squared_error(data, targets),
                self.r2_score(data, targets))

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
            print('Best estimator : ', self.__lls.best_estimator_)
        except:
            print("Linear_least_squares: __lls didn't use GridSearchCV "
                  "or RandomSearchCV.")
