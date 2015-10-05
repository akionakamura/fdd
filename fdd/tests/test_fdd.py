from unittest import TestCase
from fdd import FDD

class TestFDD(TestCase):
    def test_default_creation(self):
        my_fdd = FDD()
        self.assertTrue(my_fdd.name == 'DefaultName')

    def test_custom_creation(self):
        my_fdd = FDD('CustomName')
        self.assertTrue(my_fdd.name == 'CustomName')