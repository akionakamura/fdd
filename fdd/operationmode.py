
import numpy as np
import logging

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

class OperationMode:

    def __init__(self,
                 model,
                 kind,
                 status,
                 name='unnamed',
                 n_samples=10000,
                 confidence=0.99):
        self.model = model
        self.kind = kind
        self.status = status
        self.name = name
        self.confidence = confidence
        self.threshold = monte_carlo_threshold(self, n_samples, confidence)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def set_kind(self, kind):
        self.kind = kind

    def get_kind(self):
        return self.kind

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_threshold(self):
        return self.threshold


def monte_carlo_threshold(op, n_sample, confidence):
    logging.info('Obtaining the threshold with %.2f degre of significance for the %s operation mode.'
                 % (confidence, op.name))
    sample = op.model.sample(n_sample)
    logprob_sample, _ = op.model.score_samples(sample)
    sorted_logprob_sample = np.sort(logprob_sample)
    threshold = sorted_logprob_sample[int(np.round(n_sample*(1-confidence)))]
    return threshold
