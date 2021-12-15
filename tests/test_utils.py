import unittest
from idtpy.utils import *
import numpy.testing as npt

class TestUtils(unittest.TestCase):
    def test_sampling(self):
        values = [1, 1, -1, -1]
        n = len(values)
        func = sampling_func(values)
        for i, value in enumerate(values):
            self.assertEqual(func(i), value)
            self.assertEqual(func(i + n), value)
            self.assertEqual(func(i), sampling(i, values))

    def test_normalize(self):
        npt.assert_array_equal(normalize([1, 0, -1]), [1, 0, -1])
        npt.assert_array_equal(normalize([2, 0, -1]), [1, 0, -0.5])
        npt.assert_array_equal(normalize([-2, 0, -1]), [-1, 0, -0.5])


if __name__ == '__main__':
    unittest.main()
