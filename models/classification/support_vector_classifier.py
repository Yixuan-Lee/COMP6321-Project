from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class Support_vector_classifier:
    svm = None

    def __init__(self, c=1.0, k='rbf', g='auto', rs=0):
        self.svm = SVC(C=c, kernel=k, gamma=g, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.svm.fit(x_train, y_train)
        except:
            print("Support_vector_classifier: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.svm.predict(x_test), y_test)
        except:
            print("Support_vector_classifier: x_test or y_test may be wrong")
