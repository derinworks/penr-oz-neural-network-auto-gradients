import math
import unittest
from parameterized import parameterized
import functions as func

class TestFunctions(unittest.TestCase):

    @parameterized.expand([
        (func.sigmoid(-501), 0.0),
        (func.sigmoid(-499), 0.0),
        (func.sigmoid(-1.5), 0.1824),
        (func.sigmoid(0.0), 0.5),
        (func.sigmoid(1.5), 0.8176),
        (func.sigmoid(499), 1.0),
        (func.sigmoid(501), 1.0),
        (func.sigmoid_derivative(-0.5), -0.75),
        (func.sigmoid_derivative(0.0), 0.0),
        (func.sigmoid_derivative(0.5), 0.25),
        (func.relu(-1.5), 0.0),
        (func.relu(0.0), 0.0),
        (func.relu(1.5), 1.5),
        (func.relu_derivative(-1.5), 0.0),
        (func.relu_derivative(0.0), 0.0),
        (func.relu_derivative(1.5), 1.0),
        (func.tanh(-1.5), -0.9051),
        (func.tanh(0.0), 0.0),
        (func.tanh(1.5), 0.9051),
        (func.tanh_derivative(-0.9051), 0.1808),
        (func.tanh_derivative(0.0), 1.0),
        (func.tanh_derivative(0.9051), 0.1808),
        (func.mean_squared_error([-1, 1], [0, 1]), 0.5),
        (func.mean_squared_error([0, 0], [1, 0]), 0.5),
        (func.mean_squared_error([1.5, 1.0], [0.0, 0.7]), 1.17),
        (func.cross_entropy_loss([0.2, 0.3, 0.5], [0.3, 0.4, 0.4]), 1.2417),
        (func.cross_entropy_loss([-1.5, 0.2, 2.0], [0.3, 0.4, 0.4]), 8.9331),
        (func.cross_entropy_gradient(0.2, 0.3), 0.1),
    ])
    def test_func_scalar(self, actual: float, expected: float):
        self.assertAlmostEqual(expected, actual, 4)

    @parameterized.expand([
        (func.softmax([-1.5, 0.2, 2.0]), [0.0253, 0.1383, 0.8365]),
        (func.softmax([0.0, 0.0, 0.0]), [0.3333, 0.3333, 0.3333]),
        (func.softmax([1.5, 0.3, -2.0]), [0.7511, 0.2262, 0.0227]),
        (func.batch_norm([-1.5, 0.2, 2.0]), [-1.2129, -0.0233, 1.2362]),
        (func.batch_norm([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]),
        (func.apply_dropout([-1.5, 0.2, 2.0], 0.0), [-1.5, 0.2, 2.0]),
        (func.apply_dropout([-1.5, 0.2, 2.0], 0.99999), [0.0, 0.0, 0.0]),
        (func.apply_dropout([-1.5, 0.2, 2.0], 1.0), [-1.5, 0.2, 2.0]),
    ])
    def test_func_list(self, actual: list[float], expected: list[float]):
        self.assertEqual(len(expected), len(actual))
        for i, (e, a) in enumerate(zip(expected, actual)):
            self.assertAlmostEqual(e, a, 4, f"Element at index {i}")

    @parameterized.expand([
        (math.nan, func.tanh, ValueError),
        (math.inf, func.tanh, ValueError),
        ([math.nan, 1, -1], func.softmax, ValueError),
        ([math.inf, 1, -1], func.softmax, ValueError),
    ])
    def test_func_error(self, x, f, expected):
        with self.assertRaises(expected):
            f(x)

if __name__ == "__main__":
    unittest.main()
