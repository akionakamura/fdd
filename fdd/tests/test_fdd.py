from unittest import TestCase
from fdd import FDD
import numpy as np
from sklearn.datasets import make_blobs
from numpy.testing import assert_array_almost_equal
from scipy.stats import randint

# TODO Complete test cases.
class TestFDD(TestCase):
    def test_default_creation(self):
        my_fdd = FDD()
        self.assertTrue(my_fdd.name == 'DefaultName')
        self.assertTrue(my_fdd.training_type == 'grid')
        self.assertTrue(np.array_equal(my_fdd.n_components, np.array([1, 2])))
        self.assertTrue(np.array_equal(my_fdd.n_pc, np.array([1])))
        self.assertTrue(my_fdd.verbose == 0)
        self.assertTrue(my_fdd.n_jobs == 1)
        self.assertTrue(my_fdd.n_iter_search == 10)

    def test_custom_creation(self):
        my_fdd = FDD(name='CustomName',
                     training_type='single',
                     n_components=2,
                     n_pc=3,
                     verbose=2,
                     n_jobs=2,
                     n_iter_search=5)
        self.assertTrue(my_fdd.name == 'CustomName')
        self.assertTrue(my_fdd.training_type == 'single')
        self.assertTrue(my_fdd.n_components == 2)
        self.assertTrue(my_fdd.n_pc == 3)
        self.assertTrue(my_fdd.verbose == 2)
        self.assertTrue(my_fdd.n_jobs == 2)
        self.assertTrue(my_fdd.n_iter_search == 5)

    def test_single_train(self):
        n_samples = 10000
        n_features = 5
        centers = np.array([[10, 5, 1, -5, -10]])
        data, label = make_blobs(n_features=n_features,
                                 n_samples=n_samples,
                                 centers=centers,
                                 random_state=1)
        fdd = FDD(name='SingleFDD',
                  training_type='single',
                  n_components=1,
                  n_pc=3)
        fdd.train(data)
        trained_model = fdd.models[0]
        self.assertTrue(trained_model.get_kind() == 'normal')
        assert_array_almost_equal(trained_model.get_model().means_, centers, decimal=1)
        self.assertTrue(len(fdd.models) == 1)

    def test_random_train(self):
        n_samples = 10000
        n_features = 5
        centers = np.array([[10, 5, 1, -5, -10]])
        data, label = make_blobs(n_features=n_features,
                                 n_samples=n_samples,
                                 centers=centers,
                                 random_state=1)
        fdd = FDD(name='RandomFDD',
                  training_type='random',
                  n_components=randint(1, 3),
                  n_pc=randint(1, 3),
                  n_iter_search=5)
        fdd.train(data)
        trained_model = fdd.models[0]
        self.assertTrue(trained_model.get_kind() == 'normal')
        self.assertTrue(trained_model.get_model().means_.shape[1] == 5)
        self.assertTrue(trained_model.get_model().means_.shape[0] == 2)
        self.assertTrue(len(fdd.models) == 1)

    def test_grid_train(self):
        n_samples = 10000
        n_features = 5
        centers = np.array([[10, 5, 1, -5, -10]])
        data, label = make_blobs(n_features=n_features,
                                 n_samples=n_samples,
                                 centers=centers,
                                 random_state=1)
        fdd = FDD(name='GridFDD',
                  training_type='grid',
                  n_components=np.array([1, 2, 3]),
                  n_pc=np.array([1, 2, 3]))
        fdd.train(data)
        trained_model = fdd.models[0]
        self.assertTrue(trained_model.get_kind() == 'normal')
        self.assertTrue(trained_model.get_model().means_.shape[1] == 5)
        self.assertTrue(trained_model.get_model().means_.shape[0] == 3)
        self.assertTrue(len(fdd.models) == 1)
