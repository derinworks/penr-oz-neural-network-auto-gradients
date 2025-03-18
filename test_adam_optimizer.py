import unittest
from parameterized import parameterized
from adam_optimizer import AdamOptimizer

class TestAdamOptimizer(unittest.TestCase):

    def setUp(self):
        # Initialize optimizer
        self.optimizer = AdamOptimizer()

    def test_init(self):
        self.assertEqual(0, self.optimizer.t)
        self.assertEqual(0.0, self.optimizer.m)
        self.assertEqual(0.0, self.optimizer.v)

    @parameterized.expand([
        (1,   -500, -0.01751078),
        (500, -500, -0.12466566),
        (0,   -0.1, -0.03162277),
        (1,   -0.1, -0.01751078),
        (500, -0.1, -0.12466541),
        (0,    0.0,  0.00000000),
        (500,  0.0,  0.00000000),
        (0,    0.1,  0.03162277),
        (1,    0.1,  0.01751078),
        (500,  0.1,  0.12466541),
        (0,    0.2,  0.03162278),
        (1,    0.2,  0.017510782),
        (500,  0.2,  0.124665537),
        (0,    500,  0.031622777),
        (1,    500,  0.017510784),
        (500,  500,  0.124665661),
    ])
    def test_step(self, t: int, gradient: float, expected: float):

        self.optimizer.t = t

        step = self.optimizer.step(gradient)

        self.assertEqual(t + 1, self.optimizer.t)
        self.assertAlmostEqual(expected, step, 8)


if __name__ == '__main__':
    unittest.main()
