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
        (1,   -500, -0.07624798),
        (500, -500, -0.069674536),
        (0,   -0.1, -0.09999998),
        (1,   -0.1, -0.076247946),
        (500, -0.1, -0.0696577542),
        (0,    0.0,  0.0),
        (500,  0.0,  0.0),
        (0,    0.1,  0.09999998),
        (1,    0.1,  0.07624795),
        (500,  0.1,  0.06965775),
        (0,    0.2,  0.09999999),
        (1,    0.2,  0.07624796),
        (500,  0.2,  0.06966615),
        (0,    500,  0.1),
        (1,    500,  0.07624798),
        (500,  500,  0.06967454),
    ])
    def test_step(self, t: int, gradient: float, expected: float):
        self.optimizer.t = t

        step = self.optimizer.step(gradient)

        self.assertEqual(t + 1, self.optimizer.t)
        self.assertAlmostEqual(expected, step, 8)

if __name__ == '__main__':
    unittest.main()
