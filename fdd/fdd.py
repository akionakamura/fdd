from time import time
import logging

from sklearn.mixture import PGMM
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from pyspark import SparkConf, SparkContext
from scipy.stats import binom
import numpy as np

from operationmode import OperationMode


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
                 covar_types=np.array([7]),
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
        self.covar_types = covar_types
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter_search = n_iter_search
        self.n_samples = n_samples
        self.confidence = confidence
        self.models = []

    def train(self, data, name, kind='normal', status='OK'):
        logging.info('Training starting...')
        train_dict = {
            'random': _train_random,
            'grid': _train_grid,
            'single': _train_single,
            'spark': _train_spark}
        pgmm_model = train_dict[self.training_type](data,
                                                    self.n_components,
                                                    self.n_pc,
                                                    self.covar_types,
                                                    self.verbose,
                                                    self.n_jobs,
                                                    self.n_iter_search)
        model = OperationMode(model=pgmm_model,
                              kind=kind,
                              status=status,
                              model_id=len(self.models),
                              name=name,
                              n_samples=self.n_samples,
                              confidence=self.confidence)
        # tr = threshold_by_data(pgmm_model, data, confidence=self.confidence)
        # model.set_threshold(tr)
        self.models.append(model)

    def monitor(self, data, model_id=0):
        gamma = 0.6
        n = data.shape[0]
        # Get Operation Mode
        op_mode = self.models[model_id]
        # Compute the limit of out-of-bounds sample to be detected as out of the model.
        limit = np.round(binom.ppf(op_mode.confidence, n, 1-op_mode.confidence))
        # Compute the log likelihood.
        logprob, responsability = op_mode.model.score_samples(data)
        filtered_stats = exponential_filter(logprob, gamma)
        idx_out = -filtered_stats > op_mode.threshold
        num_out = np.sum(idx_out)
        out = num_out > limit
        data_out = data[idx_out,]
        return -filtered_stats, op_mode.threshold, out, num_out, data_out, op_mode.model_id

    def monitor_all(self, data):
        n = data.shape[0]
        num_out_best = n
        best_model_id = 0
        for model in self.models:
            _, _, out, num_out, _, _ = self.monitor(data, model.model_id)
            if num_out < num_out_best:
                num_out_best = num_out
                best_model_id = model.model_id

        return self.monitor(data, best_model_id)

    def fdd(self, data):
        if (len(self.models) <= 0):
            print('There is no model registered, creating a normal one.')
            self.train(data, 'Normal Condition', 'Normal', 'OK')
            return False
        else:
            stats, threshold, out, num_out, data_out, id = self.monitor(data, 0)
            print(num_out)
            if out:
                print('Out of normal operation condition detected.')
                stats2, threhold2, out2, num_out2, data_out2, id2 = self.monitor_all(data_out)
                print('The best fitting model found was: ' + self.models[id2].name)
                print(out2)
                print(num_out2)
                print(self.models[id2].model)
                if out2:
                    print('Unrecognized behaviour, training a new model.')
                    out_perct = float(num_out2) / len(data_out)
                    print(out_perct)
                    self.train(data_out, 'Fault'+ str(len(self.models)), 'OK')
            else:
                print('Normal operation condition detected.')

            return stats, threshold


def exponential_filter(stats, gamma):
    n = stats.shape[0]
    filtered = np.zeros(n)
    filtered[0] = stats[0]
    for i in np.arange(n-1)+1:
        filtered[i] = (1-gamma)*filtered[i-1] + gamma*stats[i]

    return filtered


def threshold_by_data(pgmm, data, confidence):
    n_samples = data.shape[0]
    logprob, _ = pgmm.score_samples(data)
    sorted_logprob = np.sort(logprob)
    threshold = sorted_logprob[int(np.round(n_samples*(1-confidence)))]
    return -threshold


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


covar_type_int = {1: 'RRR',
                  2: 'RRU',
                  3: 'RUR',
                  4: 'RUU',
                  5: 'URR',
                  6: 'URU',
                  7: 'UUR',
                  8: 'UUU',
                  }


def train_with_parameters(params, data_broadcast):
    data = data_broadcast.value
    n_components = params[0]
    n_pc = params[1]

    covariance_type = covar_type_int[params[2]]
    model = PGMM(n_components=n_components,
                 n_pc=n_pc,
                 covariance_type=covariance_type, tol=1e-6)
    model.fit(data)
    return model.bic(data), model


def _train_spark(data, n_components, n_pc, covar_types, verbose, n_jobs, n_iter_search):
    conf = (SparkConf()
             .setMaster("local[*]")
             .setAppName("FDD")
             .set("spark.executor.memory", "2g"))
    sc = SparkContext(conf=conf)

    parameters = cartesian((n_components,
                            n_pc,
                            covar_types))
    parameters_rdd = sc.parallelize(parameters, n_jobs)
    data_broadcast = sc.broadcast(data)
    models = parameters_rdd.map(lambda param: train_with_parameters(param, data_broadcast))
    sorted_models = models.sortBy(lambda model: model[0])
    best_model = sorted_models.collect()[0][1]
    sc.stop()
    return best_model


# TODO(akio) Adjust the usage of the covar_types
def _train_grid(data, n_components, n_pc, covar_types, verbose, n_jobs, n_iter_search):
    param_grid = [{'n_components': n_components,
                   'n_pc': n_pc,
                   'covariance_type': covar_types,
                   'verbose': [verbose]}]

    grid_search = GridSearchCV(PGMM(),
                               param_grid=param_grid,
                               scoring=PGMM.bic,
                               verbose=verbose,
                               n_jobs=n_jobs)
    grid_start = time()
    grid_search.fit(data)
    grid_end = time()
    logging.info('GridSearchCV took %.2f seconds for %d candidate parameter settings.'
      % (grid_end - grid_start, len(grid_search.grid_scores_)))
    return grid_search.best_estimator_


# TODO(akio) Adjust the usage of the covar_types
def _train_random(data, n_components, n_pc, covar_types, verbose, n_jobs, n_iter_search):
    param_distribution = {'n_components': n_components,
                          'n_pc': n_pc,
                          'covariance_type': covar_types,
                          'verbose': [verbose]}
    random_search = RandomizedSearchCV(PGMM(),
                                       param_distributions=param_distribution,
                                       n_iter=n_iter_search,
                                       scoring=PGMM.bic,
                                       verbose=verbose,
                                       n_jobs=n_jobs)
    random_start = time()
    random_search.fit(data)
    random_end = time()
    logging.info("RandomSearchCV took %.2f seconds for %d candidate parameter settings."
      % (random_end - random_start, len(random_search.grid_scores_)))
    return random_search.best_estimator_


# TODO(akio) Adjust the usage of the covar_types
def _train_single(data, n_components, n_pc, covar_types, verbose, n_jobs, n_iter_search):
    model = PGMM(n_components=n_components,
                 n_pc=n_pc,
                 covariance_type=covar_types,
                 verbose=verbose)
    single_start = time()
    model.fit(data)
    single_end = time()
    logging.info("Single PGMM took %.2f seconds."
      % (single_end - single_start))
    return model
