from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Logistic_regression:
    lr = None

    def __init__(self, c=1.0, d=False, p='l2', s='liblinear', rs=0):
        self.lr = LogisticRegression(C=c, dual=d, penalty=p, solver=s,
            random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.lr.fit(x_train, y_train)
        except:
            print("Logistic_regression: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.lr.predict(x_test), y_test)
        except:
            print("Logistic_regression: x_test or y_test may be wrong")
