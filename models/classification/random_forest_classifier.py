from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Random_forest_classifier:
    rfc = None

    def __init__(self, n=10, c='gini', md=None, rs=0):
        self.rfc = RandomForestClassifier(n_estimators=n, criterion=c,
            max_depth=md, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.rfc.fit(x_train, y_train)
        except:
            print("Random_forest_classifier: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.rfc.predict(x_test), y_test)
        except:
            print("Random_forest_classifier: x_test or y_test may be wrong")
