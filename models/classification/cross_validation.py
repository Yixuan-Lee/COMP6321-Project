from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Cross_validation:

    def __init__(self):
        pass

    @staticmethod
    def grid_search_cv(model, param_grid, cv, n_jobs, x_train, y_train):
        """
        apply Grid Search Cross Validation

        :param model:       model instance
        :param param_grid:  parameters grid argument given to GridSearchCV
        :param cv:          number of folds
        :param n_jobs:      number of jobs to run in parallel
        :param x_train:     training data
        :param y_train:     training targets
        :return: the best estimator
        """
        gscv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs)
        gscv.fit(x_train, y_train)

        return gscv

    @staticmethod
    def random_search_cv(model, param_dist, cv, n_iter, n_jobs, x_train, y_train):
        """
        apply Random Search Cross Validation

        :param model:       model instance
        :param param_dist:  parameters grid argument given to RandomSearchCV
        :param cv:          number of folds
        :param n_iter       number of parameter settings that are sampled
        :param n_jobs:      number of jobs to run in parallel
        :param x_train:     training data
        :param y_train:     training targets
        :return: the best estimator
        """
        rscv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            cv=cv,
            verbose=0,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=0)
        rscv.fit(x_train, y_train)

        return rscv
