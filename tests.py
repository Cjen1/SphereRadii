import numpy as np
import lib
import unittest
from scipy.sparse import dok_matrix as sparse

class TestLib(unittest.TestCase):

    def test_triangle(self):
        X = np.array([
            [[0,0,0]],
            [[1,1,0]],
            [[0,1,0]],
            ])

        pairs = lib.find_pairs_within_d(X, 2)
        self.assertEqual(pairs, [{1,2}, {0,2}, {0,1}])

        pairs = lib.find_k_nearest_pairs(X, 2)
        self.assertEqual(pairs, [{1,2}, {0,2}, {0,1}])

        pair_dists = lib.find_pair_dists(X, 2)
        self.assertEqual(list(pair_dists.items()), ([((0,1),np.sqrt(2)), ((0,2),1), ((1,2),1)]))

        radii = lib.get_radii(X,2)
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2)/2, (1-np.sqrt(2)/2)])
        diff = np.absolute(radii - expected)
        self.assertTrue(np.all(diff < 0.0001))

    def test_triangle_time(self):
        # First and last time stamps should not affect result
        X = np.array([
            [[0, 0, 0], [0,0,0], [0, 0, 0]],
            [[0, 10,0], [1,1,0], [0, 10,0]],
            [[10,0, 0], [0,1,0], [10,0, 0]]
            ])

        pairs = lib.find_pairs_within_d(X, 2)
        self.assertEqual(pairs, [{1,2}, {0,2}, {0,1}])

        pairs = lib.find_k_nearest_pairs(X, 2)
        self.assertEqual(pairs, [{1,2}, {0,2}, {0,1}])

        pair_dists = lib.find_pair_dists(X, 2)
        self.assertEqual(list(pair_dists.items()), ([((0,1),np.sqrt(2)), ((0,2),1), ((1,2),1)]))

        radii = lib.get_radii(X,2)
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2)/2, (1-np.sqrt(2)/2)])
        diff = np.absolute(radii - expected)
        self.assertTrue(np.all(diff < 0.0001))

if __name__ == '__main__':
    unittest.main()
