from sklearn.tree import DecisionTreeClassifier
from cross_validation import Cross_validation
from sklearn.metrics import accuracy_score


class Decision_tree_classifier(Cross_validation):
    __dtc = None
    __param = {}

    def __init__(self, x_train=None, y_train=None, cv=3,
            criterion=('gini',),  max_depth=(None,), min_samples_leaf=(1,),
            grid_search=False, random_search=False):

        self.__dtc = DecisionTreeClassifier(random_state=0)

        try:
            self.__param = {
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf
            }
            if grid_search and random_search:
                print('only one of GridSearch and RandomSearch can be used.')
                raise Exception
            else:
                if grid_search:
                    # apply GridSearchCV and get the best estimator
                    self.__dtc = super().grid_search_cv(self.__dtc,
                        self.__param, cv, x_train, y_train)
                elif random_search:
                    # apply RandomSearchCV and get the best estimator
                    self.__dtc = super().random_search_cv(self.__dtc,
                        self.__param, cv, x_train, y_train)
                else:
                    # fit data directly
                    self.__dtc.fit(x_train, y_train)
        except:
            print("Decision_tree_classifier: x_train or y_train may be wrong")

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
                y_pred=self.__dtc.predict(x_test))
        except:
            print("Decision_tree_classifier: x_test or y_test may be wrong")

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
            print('Best estimator : ', self.__dtc.best_estimator_)
        except:
            print("Decision_tree_classifier: __dtc didn't use GridSearchCV "
                  "or RandomSearchCV.")
