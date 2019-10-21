from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Decision_tree_classifier:
    dtc = None

    def __init__(self, c='gini', s='best', md=None, rs=0):
        self.dtc = DecisionTreeClassifier(criterion=c, splitter=s,
            max_depth=md, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.dtc.fit(x_train, y_train)
        except:
            print("Decision_tree_classifier: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.dtc.predict(x_test), y_test)
        except:
            print("Decision_tree_classifier: x_test or y_test may be wrong")

