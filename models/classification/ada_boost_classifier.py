from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


class Ada_boost_classifier:
    abc = None

    def __init__(self, n=50, lr=1, algo='SAMME.R', rs=0):
        self.abc = AdaBoostClassifier(n_estimators=n, learning_rate=lr,
            algorithm=algo, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.abc.fit(x_train, y_train)
        except:
            print("Ada_boost_classifier: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.abc.predict(x_test), y_test)
        except:
            print("Ada_boost_classifier: x_test or y_test may be wrong")
