from unittest import TestCase

import fdd

class TestJoke(TestCase):
    def test_is_string(self):
        s = fdd.joke()
        self.assertTrue(isinstance(s, basestring))