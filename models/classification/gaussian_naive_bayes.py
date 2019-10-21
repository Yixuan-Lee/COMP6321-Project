from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class Gaussian_naive_bayes:
    gnb = None

    def __init__(self, p=None, vs=1e-09):
        self.gnb = GaussianNB(priors=p, var_smoothing=vs)

    def train(self, x_train=None, y_train=None):
        try:
            self.gnb.fit(x_train, y_train)
        except:
            print("Gaussian_naive_bayes: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.gnb.predict(x_test), y_test)
        except:
            print("Gaussian_naive_bayes: x_test or y_test may be wrong")
