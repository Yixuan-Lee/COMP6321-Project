from sklearn.naive_bayes import GaussianNB
from cross_validation import Cross_validation
from sklearn.metrics import accuracy_score


class Gaussian_naive_bayes(Cross_validation):
    __gnb = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10,
            var_smoothing=(1e-9,),
            grid_search=False, random_search=False):

        self.__gnb = GaussianNB()

        try:
            self.__param = {
                'var_smoothing': var_smoothing
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__gnb = super().grid_search_cv(self.__gnb,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__gnb = super().random_search_cv(self.__gnb,
                        self.__param, cv, n_iter, x_train, y_train)
                else:
                    # fit data directly
                    self.__gnb.fit(x_train, y_train)
        except:
            print("Gaussian_naive_bayes: x_train or y_train may be wrong")

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
                y_pred=self.__gnb.predict(x_test))
        except:
            print("Gaussian_naive_bayes: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__gnb.best_estimator_)
        except:
            print("Gaussian_naive_bayes: __gnb didn't use GridSearchCV "
                  "or RandomSearchCV.")
