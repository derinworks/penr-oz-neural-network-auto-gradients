import unittest
from parameterized import parameterized
from gradients import Scalar, Vector

class TestGradients(unittest.TestCase):

    @parameterized.expand([
        (0,),
        (0.0,),
        (1,),
        (1.0,),
        (2.3,),
    ])
    def test_init_scalar(self, value):
        scalar = Scalar(value)

        self.assertEqual(value, scalar.value)
        self.assertEqual((), scalar.operands)
        self.assertEqual(0.0, scalar.gradient)
        self.assertIsNone(scalar.gradient_overall)
        self.assertIsNotNone(scalar.adam_optimizer)
        self.assertIsNone(scalar.gradient_optimized)

    @parameterized.expand([
        (Scalar(0), "Scalar(0, gradient=0.0)"),
        (Scalar(1) + 1, "Summation(2.0, gradient=0.0)"),
        (Scalar(1.1) * 2, "Multiplication(2.2, gradient=0.0)"),
        (Scalar(2)**3, "Exponentiation(8.0, gradient=0.0)"),
        (Vector([0]), "Vector([Scalar(0, gradient=0.0)])"),
    ])
    def test_repr(self, actual, expected):
        self.assertEqual(expected, str(actual))

    @parameterized.expand([
        (1, ()),
        (1.0, 1.0),
        (1.0, (.5, .5))
    ])
    def test_init_scalar_with_operands(self, value, operands):
        scalar = Scalar(value, operands)

        self.assertEqual(value, scalar.value)
        self.assertEqual(operands, scalar.operands)

    @parameterized.expand([
        ([0],),
        ([0.0],),
        ([1, 2.0],),
        ([1.0, 2.0, -1],),
        ([Scalar(-1), Scalar(1)],),
    ])
    def test_init_vector(self, values):
        vector = Vector(values)

        self.assertEqual(len(values), len(vector.scalars))

    @parameterized.expand([
        (1.0, 0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 2.0),
        (1.1, 2.3, 3.4),
        (1.1, Scalar(2.3), 3.4),
        (1.1, 2 + Scalar(0.3), 3.4),
    ])
    def test_add(self, value, addend, expected):
        result_scalar = Scalar(value) + addend

        self.assertEqual(expected, result_scalar.value)

    def test_add_wrong_type(self):
        with self.assertRaises(TypeError) as te:
            Scalar(1.0) + "b0gU2"

        self.assertEqual("b0gU2 of type str not supported", str(te.exception))

    @parameterized.expand([
        (1.0, 0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.1, 2.3, -1.2),
        (1.1, Scalar(2.3), -1.2),
        (1.1, 3 - Scalar(0.7), -1.2),
    ])
    def test_subtract(self, value, subtrahend, expected):
        result_scalar = Scalar(value) - subtrahend

        self.assertAlmostEqual(expected, result_scalar.value, 4)

    @parameterized.expand([
        (1.0, 0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.1, 2.3, 2.53),
        (1.1, Scalar(2.3), 2.53),
        (-1.1, 2, -2.2),
        (1.1, -Scalar(2.0), -2.2),
        (1.1, 2 * Scalar(1.0), 2.2),
    ])
    def test_multiply(self, value, multiplier, expected):
        result_scalar = Scalar(value) * multiplier

        self.assertEqual(expected, result_scalar.value)

    def test_multiply_wrong_type(self):
        with self.assertRaises(TypeError) as te:
            Scalar(1.0) * "b0gU2"

        self.assertEqual("b0gU2 of type str not supported", str(te.exception))

    @parameterized.expand([
        (1.0, 1, 1.0),
        (1.0, 1.0, 1.0),
        (4.0, 2, 2.0),
        (4.2, Scalar(2.0), 2.1),
        (-2.2, 2, -1.1),
        (3.3, 3 / Scalar(1.0), 1.1),
    ])
    def test_divide(self, value, multiplier, expected):
        result_scalar = Scalar(value) / multiplier

        self.assertAlmostEqual(expected, result_scalar.value, 4)

    @parameterized.expand([
        (1.0, 0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.1, 2, 1.21),
        (1.1, Scalar(2), 1.21),
    ])
    def test_exponentiation(self, value, exponent, expected):
        result_scalar = Scalar(value)**exponent

        self.assertAlmostEqual(expected, result_scalar.value, 4)

    def test_exponentiation_wrong_type(self):
        with self.assertRaises(TypeError) as te:
            Scalar(1.0)**"b0gU2"

        self.assertEqual("b0gU2 of type str not supported", str(te.exception))

    @parameterized.expand([
        ([Scalar(2), Scalar(1)], [1, 2], 4),
        ([3, 2], [Scalar(1), Scalar(2)], 7),
        ([3.1, 2.2], [Scalar(1.1), Scalar(2.2)], 8.25),
    ])
    def test_dot_product(self, values1: list, values2: list, expected):
        result_scalar = Vector(values1).dot(Vector(values2))

        self.assertAlmostEqual(expected, result_scalar.value, 4)

    @parameterized.expand([
        (1.0, "sigmoid", 0.7311),
        (-2, "relu", 0.0),
        (3.0, "tanh", 0.9951),
        (1.1, "softmax", 1.1),
    ])
    def test_scalar_activation(self, value, algo, expected):
        actual = Scalar(value).activate(algo)

        self.assertAlmostEqual(expected, actual.value, 4)

    @parameterized.expand([
        ([2.0, 1.0], "softmax", [0.7311, 0.2689]),
        ([2.0, 1.0, -1.0], "softmax", [0.7054, 0.2595, 0.0351]),
    ])
    def test_vector_activation(self, values, algo, expected):
        actual = Vector(values).activate(algo)

        self.assertEqual(len(expected), len(actual.scalars))
        for i, (e, a) in enumerate(zip(expected, actual.scalars)):
            self.assertAlmostEqual(e, a.value, 4, f"Elements at index {i}")

    def test_activation_unsupported(self):
        with self.assertRaises(ValueError) as ve:
            Scalar(1.0).activate("b0gU2")

        self.assertEqual("Unsupported activation algorithm: b0gU2", str(ve.exception))

    @parameterized.expand([
        (Vector([-1.5, 0.2, 2.0]).batch_norm(), [-1.2129, -0.0233, 1.2362]),
        (Vector([-1.5, 0.2, 2.0]).apply_dropout(0.99999), [0.0, 0.0, 0.0]),
        (Vector([-1.5, 0.2, 2.0]).apply_dropout(1.0), [-1.5, 0.2, 2.0]),
    ])
    def test_vector_apply_func(self, actual: list[float], expected: list[float]):
        self.assertEqual(len(expected), len(actual))
        for i, (e, a) in enumerate(zip(expected, actual)):
            self.assertAlmostEqual(e, a, 4, f"Element at index {i}")

    @parameterized.expand([
        (Vector([-1.5, 0.2, 2.0]), [0.0, 0.5, 1.0], 1.1133),
        (Vector([-1.5, 0.2, 2.0]).activate("softmax"), [0.2, 0.3, 0.5], 1.4186),
    ])
    def test_vector_calculate_cost(self, vector: Vector, target: list[float], expected: float):
        actual = vector.calculate_cost(Vector(target))

        self.assertAlmostEqual(expected, actual.value, 4)

    @parameterized.expand([
        (Scalar(1.0), [1.0]),
        (Scalar(1.0) + 2.0, [1.0, 1.0, 1.0]),
        (Scalar(-2.0) * 4.0, [1.0, -2.0, 4.0]),
        (Scalar(-2.0) * (Scalar(-6.0) + 10.0), [1.0, -2.0, -2.0, -2.0, 4.0]),
        (Scalar(-2.0) * (Scalar(10.0) + (Scalar(2.0) * -3)), [1.0, -2.0, -2.0, -4.0, 6.0, -2.0, 4.0]),
        (Scalar(2)**3 + 1, [1.0, 1.0, 1.0, 0.0, 12.0]),
        (Scalar(1.0).activate("sigmoid"), [1.0, 0.1966]),
        ((3 * Scalar(-2) + 1).activate("relu"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ((Scalar(2) + 1).activate("relu"), [1.0, 1.0, 1.0, 1.0]),
        (Scalar(1.0).activate("tanh"), [1.0, 0.42]),
        (Vector([-1.5, 0.2]).activate("softmax").calculate_cost(Vector([0.2, 0.8])), [1.0, 0.2922, -0.3078]),
    ])
    def test_scalar_back_propagate(self, result: Scalar, expected: list[float]):
        actual: list[Scalar] = []
        scalar_set = set()
        def dfs(scalar):
            if scalar not in scalar_set:
                scalar_set.add(scalar)
                for operand in scalar.operands:
                    dfs(operand)
                actual.append(scalar)
        dfs(result)

        result.back_propagate()

        self.assertEqual(len(expected), len(actual))
        for i, (e, a) in enumerate(zip(expected, reversed(actual))):
            self.assertAlmostEqual(e, a.gradient, 4, f"Element at index {i}")
            self.assertAlmostEqual(e, a.gradient_overall, 4, f"Element at index {i}")
            self.assertGreaterEqual(e * a.gradient_optimized, 0) # same sign

        result.clear_gradient()

        for i, (e, a) in enumerate(zip(expected, reversed(actual))):
            self.assertEqual(0.0, a.gradient, f"Element at index {i}")
            self.assertAlmostEqual(e, a.gradient_overall, 4, f"Element at index {i}")
            self.assertGreaterEqual(e * a.gradient_optimized, 0) # same sign

        result.back_propagate()

        for i, (e, a) in enumerate(zip(expected, reversed(actual))):
            self.assertAlmostEqual(e, a.gradient, 4, f"Element at index {i}")
            self.assertAlmostEqual(e, a.gradient_overall, 4, f"Element at index {i}")

if __name__ == "__main__":
    unittest.main()
