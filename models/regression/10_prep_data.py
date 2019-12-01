import os
import numpy as np

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

outfile1 = os.path.join(settings.ROOT_DIR, filepath, "Merck_Data1.npz")
outfile2 = os.path.join(settings.ROOT_DIR, filepath, "Merck_Data2.npz")

np.savez_compressed(outfile1, X=X1, y=y1)

np.savez_compressed(outfile2, X=X2, y=y2)
