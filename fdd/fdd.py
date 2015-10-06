from sklearn.mixture import MPPCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operationmode import OperationMode
import numpy as np
import logging

class FDD:
    """ Class for Fault Detection and Diagnosis

    name: string, A given name for the FDD system
    training_type: string, can be of three type:
        grid   - The Model is obtained by a Grid Search in the hyper parameter space (default)
        random - The Model is obtained by a Random Search in the hyper parameter space
        single - A single model is trained
    n_components - Number of components of the MPPCA, types varies accordingly with the training_type
        if (training_type == 'single'), n_components is simply an Integer
        if (training_type == 'random'), n_components is a integer probability density function from which
                                        the number of components will be randomized for each iteration.
        if (training_type == 'grid'), n_components is an array of integer of the desired number of components
                                      to search from. (default [1, 2])
    n_pc - Number of principal components to be used on each PPCA model, types varies accordingly with the training_type
        if (training_type == 'single'), n_pc is simply an Integer
        if (training_type == 'random'), n_pc is a integer probability density function from which
                                        the number of components will be randomized for each iteration.
        if (training_type == 'grid'), n_pc is an array of integer of the desired number of components
                                      to search from. (default [1])

    n_jobs - Number of parallel jobs running for training (default 1)
    n_iter_search - Number of iterations for the Random Serach (default 10)
    verbose - Verbosity level, 0 (default) is minimum.

    """

    def __init__(self,
                 name='DefaultName',
                 training_type='grid',
                 n_components=np.array([1, 2]),
                 n_pc=np.array([1]),
                 n_jobs=1,
                 n_iter_search=10,
                 verbose=0,
                 n_samples=10000,
                 confidence=0.99):
        logging.info('Creating FDD Class...')
        self.name = name
        self.training_type = training_type
        self.n_components = n_components
        self.n_pc = n_pc
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter_search = n_iter_search
        self.n_samples = n_samples
        self.confidence = confidence
        self.models = []

    def train(self, data):
        logging.info('Training starting...')
        train_dict = {
            'random': _train_random,
            'grid': _train_grid,
            'single': _train_single}
        mppc_model = train_dict[self.training_type](data,
                                                    self.n_components,
                                                    self.n_pc,
                                                    self.verbose,
                                                    self.n_jobs,
                                                    self.n_iter_search)
        model = OperationMode(model=mppc_model,
                              kind='normal',
                              status='OK',
                              n_samples=self.n_samples,
                              confidence=self.confidence)
        self.models.append(model)

    def dummy_function(self):
        print 'This is a dummy function.'


def _train_grid(data, n_components, n_pc, verbose, n_jobs, n_iter_search):
    param_grid = [{'n_components': n_components,
                   'n_pc': n_pc,
                   'verbose': [verbose]}]

    grid_search = GridSearchCV(MPPCA(),
                               param_grid=param_grid,
                               scoring=MPPCA.sum_score,
                               verbose=verbose,
                               n_jobs=n_jobs)
    grid_start = time()
    grid_search.fit(data)
    grid_end = time()
    logging.info('GridSearchCV took %.2f seconds for %d candidate parameter settings.'
      % (grid_end - grid_start, len(grid_search.grid_scores_)))
    return grid_search.best_estimator_


def _train_random(data, n_components, n_pc, verbose, n_jobs, n_iter_search):
    param_distribution = {'n_components': n_components,
                          'n_pc': n_pc,
                          'verbose': [verbose]}
    random_search = RandomizedSearchCV(MPPCA(),
                                       param_distributions=param_distribution,
                                       n_iter=n_iter_search,
                                       scoring=MPPCA.sum_score,
                                       verbose=verbose,
                                       n_jobs=n_jobs)
    random_start = time()
    random_search.fit(data)
    random_end = time()
    logging.info("RandomSearchCV took %.2f seconds for %d candidate parameter settings."
      % (random_end - random_start, len(random_search.grid_scores_)))
    return random_search.best_estimator_


def _train_single(data, n_components, n_pc, verbose, n_jobs, n_iter_search):
    model = MPPCA(n_components=n_components,
                  n_pc=n_pc,
                  verbose=verbose)
    single_start = time()
    model.fit(data)
    single_end = time()
    logging.info("Single MPPCA took %.2f seconds."
      % (single_end - single_start))
    return model
