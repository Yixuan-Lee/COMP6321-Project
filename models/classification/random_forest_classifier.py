from sklearn.ensemble import RandomForestClassifier
from cross_validation import Cross_validation
from sklearn.metrics import accuracy_score


class Random_forest_classifier(Cross_validation):
    __rfc = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3,
            n_estimators=(100,), criterion=('gini',), max_depth=(None,),
            grid_search=False, random_search=False):

        self.__rfc = RandomForestClassifier(random_state=0)

        try:
            self.__param = {
                'n_estimators': n_estimators,
                'criterion': criterion,
                'max_depth': max_depth
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__rfc = super().grid_search_cv(self.__rfc,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__rfc = super().random_search_cv(self.__rfc,
                        self.__param, cv, x_train, y_train)
                else:
                    # fit data directly
                    self.__rfc.fit(x_train, y_train)
        except:
            print("Random_forest_classifier: x_train or y_train may be wrong")


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
                y_pred=self.__rfc.predict(x_test))
        except:
            print("Random_forest_classifier: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__rfc.best_estimator_)
        except:
            print("Random_forest_classifier: __rfc didn't use GridSearchCV "
                  "or RandomSearchCV.")
