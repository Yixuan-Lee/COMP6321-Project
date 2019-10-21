from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class K_nearest_neighbours:
    neigh = None

    def __init__(self, n=3, w='uniform', ls=30):
        self.neigh = KNeighborsClassifier(n_neighbors=n, weights=w, leaf_size=ls)

    def train(self, x_train=None, y_train=None):
        try:
            self.neigh.fit(x_train, y_train)
        except:
            print("K_nearest_neighbours: x_train or y_train may be wrong")

    def get_accuracy(self, x_test=None, y_test=None):
        try:
            return accuracy_score(self.neigh.predict(x_test), y_test)
        except:
            print("K_nearest_neighbours: x_test or y_test may be wrong")

