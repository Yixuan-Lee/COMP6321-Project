from sklearn.linear_model import LogisticRegression
from cross_validation import Cross_validation
from sklearn.metrics import accuracy_score


class Logistic_regression(Cross_validation):
    __lr = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10,
                 C=(1.0,), penalty='l2', max_iter=100,
                 grid_search=False, random_search=False):

        self.__lr = LogisticRegression(solver='lbfgs', random_state=0)

        try:
            self.__param = {
                'C': C,
                'max_iter': max_iter,
                'penalty': penalty
            }
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
                                                         self.__param, cv, n_iter, x_train, y_train)
                else:
                    # fit data directly
                    self.__lr.fit(x_train, y_train)
        except:
            print("Logistic_regression: x_train or y_train may be wrong")

    def accuracy_score(self, x_test=None, y_test=None):
        """
        get classification accuracy score

        :param x_test: test data
        :param y_test: test targets
        :return: the accuracy score
        """
        try:
            return accuracy_score(
                y_true=y_test,
                y_pred=self.__lr.predict(x_test), )
        except:
            print("Logistic_regression: x_test or y_test may be wrong")

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
            print("Logistic_regression: __lr didn't use GridSearchCV "
                  "or RandomSearchCV.")
