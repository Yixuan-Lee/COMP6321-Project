import os
import pickle
from models import settings     # For retrieving root path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class CIFAR:
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        # read files and combines the training batches into a single giant batch
        filepath = 'datasets/cifar-10-batches-py'
        training_batches = ('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5')
        train_data = None
        train_labels = None
        for batch in training_batches:
            dict_train = self.unpickle(filepath, batch)
            data = dict_train[b'data']
            labels = dict_train[b'labels']
            if train_data is None and train_labels is None:
                train_data = data
                train_labels = labels
            else:
                train_data = np.vstack((train_data, data))
                train_labels = np.hstack((train_labels, labels))
        self.x_train = train_data
        self.y_train = train_labels
        print(self.x_train.shape)   # (50000, 3072)
        print(self.y_train.shape)   # (50000,)
        # read test_batch
        test_batch = 'test_batch'
        dict_test = self.unpickle(filepath, test_batch)
        self.x_test = dict_test[b'data']
        self.y_test = np.asarray(dict_test[b'labels'])
        print(self.x_test.shape)   # (10000, 3072)
        print(self.y_test.shape)   # (10000,)

    def unpickle(self, filepath, filename):
        with open(os.path.join(settings.ROOT_DIR, filepath, filename), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    ##################### Model training #####################
    def decision_tree_classifier(self):
        dtc = DecisionTreeClassifier(max_depth=12, random_state=0)
        dtc.fit(self.x_train, self.y_train)

        # depth / accuracy:
        # 3 / 23%
        # 4 / 25%
        # 6 / 28.12%
        # 9 / 30.41%
        # 12 / 30.45%
        print("Decision Tree classifier accuracy: %.4f %%"
              % (dtc.score(X=self.x_test, y=self.y_test) * 100))


if __name__ == '__main__':
    cifar = CIFAR()
    cifar.decision_tree_classifier()
