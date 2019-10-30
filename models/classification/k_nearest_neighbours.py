from sklearn.neighbors import KNeighborsClassifier
from cross_validation import Cross_validation
from sklearn.metrics import accuracy_score


class K_nearest_neighbours(Cross_validation):
    __neigh = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3, n_iter=10,
            n_neighbors=(3,), weights=('uniform',),
            grid_search=False, random_search=False):
        """
        K nearest neighbours constructor

        :param x_train:         training data
        :param y_train:         training targets
        :param cv:              number of fold
        :param n_neighbors:     n_neighbors paramters
        :param grid_search:     whether doing grid search
        :param random_search:   whether doing random search
        """

        self.__neigh = KNeighborsClassifier()

        try:
            self.__param = {
                'n_neighbors': n_neighbors,
                'weights': weights
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__neigh = super().grid_search_cv(self.__neigh,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__neigh = super().random_search_cv(self.__neigh,
                        self.__param, cv, n_iter, x_train, y_train)
                else:
                    # fit data directly
                    self.__neigh.fit(x_train, y_train)
        except:
            print("K_nearest_neighbours: x_train or y_train may be wrong")

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
                y_pred=self.__neigh.predict(x_test))
        except:
            print("K_nearest_neighbours: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__neigh.best_estimator_)
        except:
            print("K_nearest_neighbours: __neigh didn't use GridSearchCV "
                  "or RandomSearchCV.")
