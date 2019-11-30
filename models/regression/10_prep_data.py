import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from models import settings

filepath = 'datasets/regression_datasets/10_Merck_Molecular_Activity_Challenge'

MERCK_FILE1 = 'ACT2_competition_training.csv'
MERCK_FILE2 = 'ACT4_competition_training.csv'

MERCK_FILE1 = os.path.join(settings.ROOT_DIR, filepath, MERCK_FILE1)
MERCK_FILE2 = os.path.join(settings.ROOT_DIR, filepath, MERCK_FILE2)

# read data from the source file

with open(os.path.join(MERCK_FILE1)) as f:
    cols1 = f.readline().rstrip('\n').split(',')
with open(os.path.join(MERCK_FILE2)) as f:
    cols2 = f.readline().rstrip('\n').split(',')

# Load the actual data, ignoring first column and using second column as targets.
X1 = np.loadtxt(MERCK_FILE1, delimiter=',', usecols=range(2, len(cols1)), skiprows=1, dtype=np.uint8)
y1 = np.loadtxt(MERCK_FILE1, delimiter=',', usecols=[1], skiprows=1)
X2 = np.loadtxt(MERCK_FILE2, delimiter=',', usecols=range(2, len(cols2)), skiprows=1, dtype=np.uint8)
y2 = np.loadtxt(MERCK_FILE2, delimiter=',', usecols=[1], skiprows=1)

# separate into train and test sets
x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(X1, y1, test_size=0.33,
                     random_state=0)
# print(self.x_train[:10],self.y_train[:10])

# normalize the training set
scaler = sklearn.preprocessing.StandardScaler().fit(x_train1)
x_train1 = scaler.transform(x_train1)
# normalize the test set with the train-set mean and std
x_test1 = scaler.transform(x_test1)

# separate into train and test sets
x_train2, x_test2, y_train2, y_test2 = \
    train_test_split(X2, y2, test_size=0.33,
                     random_state=0)
# print(self.x_train[:10],self.y_train[:10])

# normalize the training set
scaler = sklearn.preprocessing.StandardScaler().fit(x_train2)
x_train2 = scaler.transform(x_train2)
# normalize the test set with the train-set mean and std
x_test2 = scaler.transform(x_test2)

outfile1 = os.path.join(settings.ROOT_DIR, filepath, "Merck_Data1.npz")
outfile2 = os.path.join(settings.ROOT_DIR, filepath, "Merck_Data2.npz")

np.savez_compressed(outfile1, x_train1=x_train1, x_test1=x_test1, y_train1=y_train1, y_test1=y_test1)

np.savez_compressed(outfile2, x_train2=x_train2, x_test2=x_test2, y_train2=y_train2, y_test2=y_test2)
