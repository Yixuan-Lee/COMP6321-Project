from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class Decision_tree_regressor:
    dtr = None

    def __init__(self, c='mse', s='best', md=None, rs=0):
        self.dtr = DecisionTreeRegressor(criterion=c, splitter=s,
            max_depth=md, random_state=rs)

    def train(self, x_train=None, y_train=None):
        try:
            self.dtr.fit(x_train, y_train)
        except:
            print("Decision_tree_regressor: x_train or y_train may be wrong")

    def get_mse(self, x_test=None, y_test=None):
        try:
            return mean_squared_error(self.dtr.predict(x_test), y_test)
        except:
            print("Decision_tree_regressor: x_test or y_test may be wrong")
