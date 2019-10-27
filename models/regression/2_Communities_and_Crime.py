import os
import numpy as np
import pandas as pd
from models import settings
from support_vector_regressor import Support_vector_regressor
from decision_tree_regressor import Decision_tree_regressor
from random_forest_regressor import Random_forest_regressor
from ada_boost_regressor import Ada_boost_regressor
from gaussian_process_regressor import Gaussian_process_regressor
from linear_regression import Linear_regression
from neural_network_regressor import Neural_network_regressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import preprocessing


class Communities_and_crime:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/2_Communities_and_Crime'
        filename = 'communities.data'

        # read the data file
        # the first 5 columns are not predictive, so ignore
        total_cols = 128
        used_cols = range(5, total_cols)
        f = pd.read_csv(os.path.join(settings.ROOT_DIR, filepath, filename),
            delimiter=',', usecols=used_cols, header=None, na_values='?',
            dtype=np.float32)   # (1994, 128)
        self.data = f.loc[:, f.columns != total_cols - 1]   # (1994, 122)
        self.targets = f.loc[:, total_cols - 1]  # (1994,)

        # pre-processing strategy 1: ignore the missing-value rows
#         self.data = self.missing_rows_with_missing_values_ignore(f)
        # pre-processing strategy 2: impute the missing-value rows with the mean
        self.data = self.missing_rows_with_univariage_feature_imputation(self.data)

        # split into the train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                random_state=0)

        # normalize the training set
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        # normalize the test set with the train-set mean and std
        self.x_test = self.scaler.transform(self.x_test)

    ##################### Data pre-processing #####################
    def missing_rows_with_missing_values_ignore(self, data):
        # drop the rows with NANs
        return data.dropna()

    def missing_rows_with_univariage_feature_imputation(self, data):
        # train an imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(data)
        # impute the missing values with the mean
        return imp.transform(data)

    ##################### Model training #####################
    def support_vector_regression(self):
        svr = Support_vector_regressor()
        svr.train(self.x_train, self.y_train)
        return svr.get_mse(self.x_test, self.y_test)

    def decision_tree_regression(self):
        dtf = Decision_tree_regressor()
        dtf.train(self.x_train, self.y_train)
        return dtf.get_mse(self.x_test, self.y_test)

    def random_forest_regression(self):
        rfr = Random_forest_regressor()
        rfr.train(self.x_train, self.y_train)
        return rfr.get_mse(self.x_test, self.y_test)

    def ada_boost_regression(self):
        abr = Ada_boost_regressor()
        abr.train(self.x_train, self.y_train)
        return abr.get_mse(self.x_test, self.y_test)

    def gaussian_process_regression(self):
        kernel = DotProduct() + WhiteKernel()
        gpr = Gaussian_process_regressor(k=kernel)
        gpr.train(self.x_train, self.y_train)
        return gpr.get_mse(self.x_test, self.y_test)

    def linear_regression(self):
        lr = Linear_regression()
        lr.train(self.x_train, self.y_train)
        return lr.get_mse(self.x_test, self.y_test)

    def neural_network_regression(self):
        nnr = Neural_network_regressor(hls=(15,), s='lbfgs')
        nnr.train(self.x_train, self.y_train)
        return nnr.get_mse(self.x_test, self.y_test)


if __name__ == '__main__':
    cac = Communities_and_crime()
    print('SVR: %.5f' % cac.support_vector_regression())
    print('DTR: %.5f' % cac.decision_tree_regression())
    print('RFR: %.5f' % cac.random_forest_regression())
    print('ABR: %.5f' % cac.ada_boost_regression())
    print('GPR: %.5f' % cac.gaussian_process_regression())
    print(' LR: %.5f' % cac.linear_regression())
    print('NNR: %.5f' % cac.neural_network_regression())
