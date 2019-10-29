from sklearn.ensemble import AdaBoostRegressor
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error


class Ada_boost_regressor(Cross_validation):
    __abr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3,
            n_estimators=(50,), learning_rate=(1.),
            grid_search=False, random_search=False):

        self.__abr = AdaBoostRegressor(random_state=0)

        try:
            self.__param = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__abr = super().grid_search_cv(self.__abr,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__abr = super().random_search_cv(self.__abr,
                        self.__param, cv, x_train, y_train)
                else:
                    # fit data directly
                    self.__abr.fit(x_train, y_train)
        except:
            print("Ada_boost_regressor: x_train or y_train may be wrong")

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
                y_pred=self.__abr.predict(x_test))
        except:
            print("Ada_boost_regressor: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__abr.best_estimator_)
        except:
            print("Ada_boost_regressor: __abr didn't use GridSearchCV "
                  "or RandomSearchCV.")
