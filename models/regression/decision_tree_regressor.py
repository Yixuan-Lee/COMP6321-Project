from sklearn.tree import DecisionTreeRegressor
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error, r2_score


class Decision_tree_regressor(Cross_validation):
    __dtr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10, n_jobs=None, scoring=None,
                 max_depth=(None,), min_samples_leaf=(1,),
                 grid_search=False, random_search=False):

        self.__dtr = DecisionTreeRegressor(random_state=0)

        try:
            self.__param = {
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__dtr = super().grid_search_cv(self.__dtr,
                                                        self.__param, cv, n_jobs, x_train, y_train, scoring=scoring)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__dtr = super().random_search_cv(self.__dtr,
                                                          self.__param, cv, n_iter, n_jobs, x_train, y_train,
                                                          scoring=scoring)
                else:
                    # fit data directly
                    self.__dtr.fit(x_train, y_train)
        except:
            print("Decision_tree_regressor: x_train or y_train may be wrong")

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
                y_pred=self.__dtr.predict(x_test))
        except:
            print("Decision_tree_regressor: x_test or y_test may be wrong")

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
                y_pred=self.__dtr.predict(x_test))
        except:
            print("Decision_tree_regressor: x_test or y_test may be wrong")

    def evaluate(self, data=None, targets=None):
        """
        evaluate the model

        :param data: training or testing data
        :param targets: targets

        :return: return (mean_square_error, r2_score)
        """
        return (self.mean_squared_error(data, targets),
                self.r2_score(data, targets))

    def predict(self, data=None):
        """
        evaluate the model by using unique evaluation function

        :param data: training or testing data
        :return: prediction
        """

        return self.__dtr.best_estimator_.predict(data)

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
            print('Best estimator : ', self.__dtr.best_estimator_)
        except:
            print("Decision_tree_regressor: __dtr didn't use GridSearchCV "
                  "or RandomSearchCV.")
