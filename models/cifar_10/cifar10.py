import os
import pickle
from models import settings     # For retrieving root path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score


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
        self.x_train = train_data       # (50000, 3072)
        self.y_train = train_labels     # (50000,)
        # read test_batch
        test_batch = 'test_batch'
        dict_test = self.unpickle(filepath, test_batch)
        self.x_test = dict_test[b'data']                # (10000, 3072)
        self.y_test = np.asarray(dict_test[b'labels'])  # (10000,)

    def unpickle(self, filepath, filename):
        with open(os.path.join(settings.ROOT_DIR, filepath, filename), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    ##################### Model training #####################
    def decision_tree_classifier(self):
        # normalize the training and testing pixels
        train_data = self.x_train / 255.0
        test_data = self.x_test / 255.0

        # train a decision tree classifier and cross validate
        dtc = DecisionTreeClassifier(random_state=0)
        max_depth = np.logspace(start=1, stop=1, base=2, num=1, dtype=np.int)
        params = {
            'max_depth': max_depth,
            'criterion': ('gini', 'entropy')
        }
        gscv = GridSearchCV(
            estimator=dtc,
            cv=3,
            n_jobs=2,
            param_grid=params)
        gscv.fit(train_data, self.y_train)

        # print the results
        print("Decision Tree classifier: ")
        print("accuracy: %.4f %%" % (accuracy_score(
            y_true=self.y_test,
            y_pred=gscv.predict(test_data)) * 100))
        print("(average='micro') recall: %.4f" % (recall_score(
            y_true=self.y_test,
            y_pred=gscv.predict(test_data),
            average='micro')))
        print("(average='weighted') recall: %.4f" % (recall_score(
            y_true=self.y_test,
            y_pred=gscv.predict(test_data),
            average='weighted')))

        # plot the decision tree
        plt.figure(figsize=(15, 15))
        plot_tree(
            decision_tree=gscv.best_estimator_,
            max_depth=4)    # only plot the top 4 layers
        plt.title('decision tree classifier (depth = ' + str(max_depth) + ')')
        plt.show()


if __name__ == '__main__':
    cifar = CIFAR()
    cifar.decision_tree_classifier()
