from sklearn.ensemble import RandomForestRegressor
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error, r2_score


class Random_forest_regressor(Cross_validation):
    __rfr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10, n_jobs=None,
            n_estimators=(10,), max_depth=(None,),
            grid_search=False, random_search=False):

        self.__rfr = RandomForestRegressor(random_state=0)

        try:
            self.__param = {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__rfr = super().grid_search_cv(self.__rfr,
                        self.__param, cv, n_jobs, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__rfr = super().random_search_cv(self.__rfr,
                        self.__param, cv, n_iter, n_jobs, x_train, y_train)
                else:
                    # fit data directly
                    self.__rfr.fit(x_train, y_train)
        except:
            print("Random_forest_regressor: x_train or y_train may be wrong")

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
                y_pred=self.__rfr.predict(x_test))
        except:
            print("Random_forest_regressor: x_test or y_test may be wrong")

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
                y_pred=self.__rfr.predict(x_test))
        except:
            print("Random_forest_regressor: x_test or y_test may be wrong")

    def evaluate(self, data=None, targets=None):
        """
        evaluate the model

        :param data: training or testing data
        :param targets: targets

        :return: return (mean_square_error, r2_score)
        """
        return (self.mean_squared_error(data, targets),
                self.r2_score(data,targets))

    def predict(self, data=None):
        """
        evaluate the model by using unique evaluation function

        :param data: training or testing data
        :return: prediction
        """

        return self.__svr.best_estimator_.predict(data)

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
            print('Best estimator : ', self.__rfr.best_estimator_)
        except:
            print("Random_forest_regressor: __rfr didn't use GridSearchCV "
                  "or RandomSearchCV.")
