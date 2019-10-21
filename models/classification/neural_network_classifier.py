from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class Neural_network_classifier:
    nnc = None

    def __init__(self, hls=(100,), a='relu', s='adam', alp=1e-4, rs=0):
        self.nnc = MLPClassifier(hidden_layer_sizes=hls, activation=a,
            solver=s, alpha=alp, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.nnc.fit(x_train, y_train)
        except:
            print("Neural_network_classifier: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.nnc.predict(x_test), y_test)
        except:
            print("Neural_network_classifier: x_test or y_test may be wrong")
