from sklearn.linear_model import LinearRegression
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error


class Linear_regression(Cross_validation):
    __lr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3,
        grid_search=False, random_search=False):

        self.__lr = LinearRegression()

        try:
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__lr = super().grid_search_cv(self.__lr,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__lr = super().random_search_cv(self.__lr,
                        self.__param, cv, x_train, y_train)
                else:
                    # fit data directly
                    self.__lr.fit(x_train, y_train)
        except:
            print("Linear_regression: x_train or y_train may be wrong")

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
                y_pred=self.__lr.predict(x_test))
        except:
            print("Linear_regression: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__lr.best_estimator_)
        except:
            print("Linear_regression: __lr didn't use GridSearchCV "
                  "or RandomSearchCV.")
