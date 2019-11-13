from sklearn.svm import SVR
from cross_validation import Cross_validation
from sklearn.metrics import mean_squared_error, r2_score


class Support_vector_regressor(Cross_validation):
    __svr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10, n_jobs=None,
            C=(1.0,), kernel=('rbf',), gamma=('auto',), coef0=(0.0,), epsilon=(0.1,),
            grid_search=False, random_search=False):

        self.__svr = SVR()

        try:
            self.__param = {
                'C': C,
                'kernel': kernel,
                'gamma': gamma,
                'coef0': coef0,
                'epsilon': epsilon
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__svr = super().grid_search_cv(self.__svr,
                        self.__param, cv, n_jobs, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__svr = super().random_search_cv(self.__svr,
                        self.__param, cv, n_iter, n_jobs, x_train, y_train)
                else:
                    # fit data directly
                    self.__svr.fit(x_train, y_train)
        except:
            print("Support_vector_regressor: x_train or y_train may be wrong")

    def mean_sqaured_error(self, x_test=None, y_test=None):
        """
        get regression mean squared error

        :param x_test: test data
        :param y_test: test targets
        :return: the accuracy score
        """
        try:
            return mean_squared_error(
                y_true=y_test,
                y_pred=self.__svr.predict(x_test))
        except:
            print("Support_vector_regressor: x_test or y_test may be wrong")

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
                y_pred=self.__svr.predict(x_test))
        except:
            print("Support_vector_regressor: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__svr.best_estimator_)
        except:
            print("Support_vector_regressor: __svr didn't use GridSearchCV "
                  "or RandomSearchCV.")
