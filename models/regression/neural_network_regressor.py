from sklearn.neural_network import MLPRegressor
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error


class Neural_network_regressor(Cross_validation):
    __nnr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3,
            hidden_layer_sizes=(100,), activation=('relu',), max_iter=(200,),
            grid_search=False, random_search=False):

        self.__nnr = MLPRegressor(random_state=0)

        try:
            self.__param = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'max_iter': max_iter
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__nnr = super().grid_search_cv(self.__nnr,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__nnr = super().random_search_cv(self.__nnr,
                        self.__param, cv, x_train, y_train)
                else:
                    # fit data directly
                    self.__nnr.fit(x_train, y_train)
        except:
            print("Neural_network_regressor: x_train or y_train may be wrong")

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
                y_pred=self.__nnr.predict(x_test))
        except:
            print("Neural_network_regressor: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__nnr.best_estimator_)
        except:
            print("Neural_network_regressor: __nnr didn't use GridSearchCV "
                  "or RandomSearchCV.")
