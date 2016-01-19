from time import time
import logging

from sklearn.mixture import PGMM
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from pyspark import SparkConf, SparkContext, StorageLevel
from scipy.stats import binom
import numpy as np
import math

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

    def set_op_name(self, id, name):
        self.models[id].set_name(name)

    def set_op_status(self, id, status):
        self.models[id].set_status(status)

    def set_op_kind(self, id, kind):
        self.models[id].set_kind(kind)

    def get_op(self, id):
        return self.models[id]

    # Get the best model accordingly with the selected training method.
    def train(self, data, name='unnamed', kind='undefined', status='pending_eval'):
        logging.info('Training starting...')
        # Created a dict with the several training methods.
        train_dict = {
            'random': _train_random,
            'grid': _train_grid,
            'single': _train_single,
            'spark': _train_spark}
        # Actually trains the model.
        pgmm_model = train_dict[self.training_type](data,
                                                    self.n_components,
                                                    self.n_pc,
                                                    self.covar_types,
                                                    self.verbose,
                                                    self.n_jobs,
                                                    self.n_iter_search)
        if pgmm_model is None:
            return None
        else:
            # If model training is succesful, add it to a OperationMode.
            model = OperationMode(model=pgmm_model,
                                  kind=kind,
                                  status=status,
                                  model_id=len(self.models),
                                  name=name,
                                  n_samples=self.n_samples,
                                  confidence=self.confidence)
            # Here we decided to override the threshold value by its value
            # computed using the training data.
            tr = threshold_by_data(pgmm_model, data, confidence=self.confidence)
            model.set_threshold(tr)

            # Append the operation mode to the know list of behaviours.
            self.models.append(model)
            return model.model_id

    # Computes the statistics and its related info of the data and
    # a selected operation model (default is 0, the normal one).
    def monitor(self, data, model_id=0):
        gamma = 0.4   # Filter rate.
        n = data.shape[0]
        # Get Operation Mode
        op_mode = self.models[model_id]
        # Compute the limit of out-of-bounds sample to be detected as out of the model.
        limit = np.round(binom.ppf(op_mode.confidence, n, 1-op_mode.confidence))
        # Compute the log likelihood.
        logprob, responsability = op_mode.model.score_samples(data)
        # Filter statistics.
        filtered_stats = exponential_filter(logprob, gamma)
        # Other info.
        idx_out = -filtered_stats > op_mode.threshold
        num_out = np.sum(idx_out)
        out = num_out > limit
        data_out = data[idx_out,]
        # Return:
        # Monitored Statistics (filtered negative log-likelihood)
        # Threshold of the selected operation model.
        # Bit indicating whether the behaviour is out of the OP.
        # Number of samples beyond the threshold.
        # Data points that were out of the model.
        # Idx of the operation mode.
        return -filtered_stats, op_mode.threshold, out, num_out, data_out, model_id

    # Find the statistics and related info of the data with regard to the best
    # fitting known OP.
    def monitor_all(self, data):
        n = data.shape[0]
        best_stats = np.zeros(n)
        best_threshold = 1e9
        best_out = False
        best_num_out = n+1
        best_data_out = np.zeros(data.shape)
        best_model_id = 0
        # Loop through every known OP.
        for model in self.models:
            stats, threshold, out, num_out, data_out, id = self.monitor(data, model.model_id)
            # If better, update info.
            if num_out < best_num_out:
                best_stats = stats
                best_threshold = threshold
                best_out = out
                best_num_out = num_out
                best_data_out = data_out
                best_model_id = model.model_id
        # Same return data of the monitor function.
        return best_stats, best_threshold, best_out, best_num_out, best_data_out, best_model_id

    # This is the main method of the FDD system, from calling the method passing the data to
    # be analysed, the algorithm should detected abnormality, diagnose if known or train new model if not.
    # Same return data of the monitor function.
    def fdd(self, data):
        # If there is no model known yet, we train one with the complete data
        # and consider it as normal behaviour.
        if len(self.models) <= 0:
            print('There is no model registered, creating a normal one.')
            new_id = self.train(data, 'Normal Condition', 'Normal', 'OK')
            if new_id is None:
                is_new = False
            else:
                is_new = True
            return [], [], [], [], new_id, is_new
        # If there is alreay some known OP, we should run the data through the models.
        else:
            # Run the data through the normal OP.
            stats_normal, threshold_normal, out_normal, num_out_normal, data_out_normal, _ = self.monitor(data)
            # If abnormal:
            if out_normal:
                # Now we should monitor the data related to all other known OP and find the
                # best fitting model.
                print('Out of normal operation condition detected.')
                stats, threshold, out, num_out, data_out, id = self.monitor_all(data_out_normal)
                # If the best fitting model is detected as out of control, we train a new model.
                if out:
                    print('Unrecognized behaviour, training a new model.')
                    print(num_out)
                    # Train the new model only with the data found as out of the normal behaviour.
                    new_id = self.train(data_out_normal)
                    # If training failed.
                    if new_id is None:
                        return stats, threshold, [], [], new_id, False
                    # If training was successful, collect the stats and info and return.
                    else:
                        new_stats, new_threshold, new_out, new_num_out, new_data_out, new_id = self.monitor(data, new_id)
                        return stats_normal, threshold_normal, new_stats, new_threshold, new_id, True
                # If is not out of control, we know this behaviour and can diagnose it.
                else:
                    print('Operation mode classified as: ' + self.models[id].name)
                    fault_stats, fault_threshold, fault_out, fault_num_out, fault_data_out, fault_id = self.monitor(data, id)
                    return stats_normal, threshold_normal, fault_stats, fault_threshold, fault_id, False
            # If normal behaviour detected, return the stats and info related to the normal OP.
            else:
                print('Normal operation condition detected.')
                return stats_normal, threshold_normal, [], [], self.models[0].model_id, False


# Exponential filter function.
def exponential_filter(stats, gamma):
    n = stats.shape[0]
    filtered = np.zeros(n)
    filtered[0] = stats[0]
    for i in np.arange(n-1)+1:
        filtered[i] = (1-gamma)*filtered[i-1] + gamma*stats[i]

    return filtered


# Compute the threshold using the training data.
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
    # We return the vector randomly permuted to balance the work.
    return np.random.permutation(out)


covar_type_int = {1: 'RRR',
                  2: 'RRU',
                  3: 'RUR',
                  4: 'RUU',
                  5: 'URR',
                  6: 'URU',
                  7: 'UUR',
                  8: 'UUU',
                  }


# Auxiliar function to the a model with specifics parameters.
def train_with_parameters(params, data_broadcast):
    data = data_broadcast.value
    n_components = params[0]
    n_pc = params[1]

    covariance_type = covar_type_int[params[2]]
    model = PGMM(n_components=n_components,
                 n_pc=n_pc,
                 covariance_type=covariance_type, tol=1e-6)
    try:
        model.fit(data)
        return model.bic(data), model
    except:
        # If failed, return the BIC as infite so the model will be discarted.
        return float("inf"), None


# Train model using spark.
def _train_spark(data, n_components, n_pc, covar_types, verbose, n_jobs, n_iter_search):
    # Spark configuration.
    conf = (SparkConf()
             .setMaster("local[" + str(n_jobs) + "]")
             .setAppName("FDD")
             .set("spark.executor.memory", "512mb")
             .set("spark.cores.max", str(n_jobs)))
    sc = SparkContext(conf=conf)
    # Build hyperparameter vectors.
    parameters = cartesian((n_components,
                            n_pc,
                            covar_types))
    # Distribute the hyperparameters vector.
    parameters_rdd = sc.parallelize(parameters, 96)
    # Broadcast the data to all workers.
    data_broadcast = sc.broadcast(data)
    # Train a model for each hyperparameter set.
    models = parameters_rdd.map(lambda param: train_with_parameters(param, data_broadcast))
    # Persist the models the avoid re-computation.
    models.persist(StorageLevel(True, True, False, True, 1))
    # Sort by BIC.
    sorted_models = models.sortBy(lambda model: model[0])
    # The first is the best model.
    best_model = sorted_models.collect()[0][1]
    sc.stop()
    return best_model


# TODO Adjust the usage of the covar_types
# Get the best model using the scikit-learn GridSearchCV.
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


# TODO Adjust the usage of the covar_types
# Get the best model using the scikit-learn RandomizedSearchCV.
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


# TODO Adjust the usage of the covar_types
# Fit a model using specifics parameters.
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
