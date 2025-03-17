import functions as func
from adam_optimizer import AdamOptimizer
from typing import Tuple

class Scalar:
    def __init__(self, value: float, operands = ()):
        """
        Initialize a scalar value with previous math operands
        :param value: a float value
        :param operands: tuple of previous math operand scalars
        """
        self.value = value
        self.operands: Tuple[Scalar, ...] = operands
        self.gradient = 0.0
        self.visited = False
        self.gradient_overall = None
        self.gradient_optimized = None
        self.adam_optimizer = AdamOptimizer()

    def __repr__(self):
        return f"{type(self).__name__}({self.value}, gradient={self.gradient})"

    @classmethod
    def _wrap(cls, other):
        if isinstance(other, float):
            return Scalar(other)
        if isinstance(other, int):
            return Scalar(float(other))
        if isinstance(other, Scalar):
            return other
        raise TypeError(f"{other} of type {type(other).__name__} not supported")

    def __add__(self, addend):
        return Summation(self, self._wrap(addend))

    def __mul__(self, multiplicand):
        return Multiplication(self, self._wrap(multiplicand))

    def __pow__(self, exponent):
        return Exponentiation(self, self._wrap(exponent))

    def __neg__(self):
        return self * -1

    def __radd__(self, addend):
        return self + addend

    def __sub__(self, subtrahend):
        return self + (-subtrahend)

    def __rsub__(self, minuend):
        return minuend + (-self)

    def __rmul__(self, multiplier):
        return self * multiplier

    def __truediv__(self, divisor):
        return self * divisor**-1

    def __rtruediv__(self, dividend):
        return dividend * self**1

    def activate(self, algo: str):
        """
        Apply activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
        :return: Activated scalar.
        """
        if algo == "sigmoid":
            return SigmoidActivation(self)
        elif algo == "relu":
            return ReluActivation(self)
        elif algo == "tanh":
            return TanhActivation(self)
        elif algo == "softmax":
            return self # softmax activation is done at vector level instead
        else:
            raise ValueError(f"Unsupported activation algorithm: {algo}")

    def _compute_gradient(self):
        pass

    def _aggregate_gradient(self, alpha: float):
        self.gradient_overall = alpha * (self.gradient_overall or self.gradient) + (1 - alpha) * self.gradient

    def _optimize_gradient(self, learning_rate: float):
        self.gradient_optimized = self.adam_optimizer.step(self.gradient_overall, learning_rate)

    def _back_propagate(self, alpha: float, learning_rate: float):
        if not self.visited:
            self.visited = True
            self._compute_gradient()
            self._aggregate_gradient(alpha)
            self._optimize_gradient(learning_rate)
            for operand in self.operands:
                operand._back_propagate(alpha, learning_rate)

    def back_propagate(self, alpha = 0.5, learning_rate = 0.1):
        """
        Applies back propagation to compute gradient of this and the previous operand scalars
        :param alpha: contribution factor of the current gradient to overall
        :param learning_rate: Learning rate for gradient descent.
        """
        self.gradient = 1.0
        self._back_propagate(alpha, learning_rate)

    def clear_gradient(self):
        if self.visited or self.gradient != 0.0:
            self.visited = False
            self.gradient = 0.0
            for operand in self.operands:
                operand.clear_gradient()

class Summation(Scalar):
    def __init__(self, left_addend: Scalar, right_addend: Scalar):
        """
        Initialize a summation result scalar with left and right addend scalars
        :param left_addend: left addend scalar
        :param right_addend: right addend scalar
        """
        super().__init__(left_addend.value + right_addend.value, (left_addend, right_addend))

    def _compute_gradient(self):
        for addend in self.operands:
            addend.gradient += self.gradient

class Multiplication(Scalar):
    def __init__(self, multiplicand: Scalar, multiplier: Scalar):
        """
        Initialize a multiplication result scalar with multiplicand and multiplier scalars
        :param multiplicand: scalar being multiplied
        :param multiplier: scalar multiplying
        """
        super().__init__(multiplicand.value * multiplier.value, (multiplicand, multiplier))

    def _compute_gradient(self):
        multiplicand, multiplier = self.operands
        multiplicand.gradient += multiplier.value * self.gradient
        multiplier.gradient += multiplicand.value * self.gradient

class Exponentiation(Scalar):
    def __init__(self, base: Scalar, exponent: Scalar):
        """
        Initialize an exponentiation result scalar with base and exponent scalars
        :param base: base scalar
        :param exponent: exponent scalar
        """
        super().__init__(base.value**exponent.value, (base, exponent))

    def _compute_gradient(self):
        base, exponent = self.operands
        base.gradient += (exponent.value * base.value**(exponent.value - 1)) * self.gradient

class SigmoidActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a sigmoid activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.sigmoid(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        pre_activation, = self.operands
        pre_activation.gradient += func.sigmoid_derivative(self.value)

class ReluActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a ReLu activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.relu(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        pre_activation, = self.operands
        pre_activation.gradient += func.relu_derivative(self.value)

class TanhActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a tanh activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.tanh(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        pre_activation, = self.operands
        pre_activation.gradient += func.tanh_derivative(self.value)

class CrossEntropyLoss(Scalar):
    def __init__(self, prediction: list[Scalar], target: list[float]):
        super().__init__(func.cross_entropy_loss([p.value for p in prediction], target), tuple(prediction))
        self.target = target

    def _compute_gradient(self):
        for p, t in zip(self.operands, self.target):
            p.gradient += func.cross_entropy_gradient(self.value, t)

class Vector:
    def __init__(self, values: list[Scalar | float]):
        """
        Initialize a vector with values
        :param values: scalars or floats populating the vector
        """
        self.scalars = [v if isinstance(v, Scalar) else Scalar(v) for v in values]

    def __repr__(self):
        return f"{type(self).__name__}({self.scalars})"

    @property
    def floats(self) -> list[float]:
        return [s.value for s in self.scalars]

    def clear_gradients(self):
        """
        Clears gradients that were populated after a back propagation run
        """
        for scalar in self.scalars:
            scalar.clear_gradient()

    def dot(self, other):
        """
        Takes dot product of this vector with another one
        :param other: another vector
        :return: scalar value of the dot product
        """
        return sum(v1 * v2 for v1, v2 in zip(self.scalars, other.scalars))

    def activate(self, algo: str):
        """
        Activates vector values
        :param algo: The activation algorithm ("softmax").
        :return: vector with activated values
        """
        if algo == "softmax":
            return SoftmaxActivation(self.scalars)
        else:
            return Vector([scalar.activate(algo) for scalar in self.scalars])

    def _apply_func_inplace(self, f) -> list[float]:
        """
        Applies function to the values of this vector in-place
        """
        result_floats = f(self.floats)
        for scalar, result_float in zip(self.scalars, result_floats):
            scalar.value = result_float
        return result_floats

    def batch_norm(self) -> list[float]:
        """
        Applies batch normalization to the values of this vector
        """
        return self._apply_func_inplace(func.batch_norm)

    def apply_dropout(self, rate: float) -> list[float]:
        """
        Drops out values of this vector by given rate
        :param rate: drop out rate
        """
        return self._apply_func_inplace(lambda floats: func.apply_dropout(floats, rate))

    def calculate_cost(self, target) -> Scalar:
        """
        Calculates cost between this vector and target
        :param target: vector
        :return: cost between this and target
        """
        return func.mean_squared_error(self.scalars, target.scalars)

class SoftmaxActivation(Vector):
    def __init__(self, scalars: list[Scalar]):
        """
        Initializes a softmax activation vector for given values
        :param scalars: scalars to be activated
        """
        super().__init__(scalars)
        softmax_values = func.softmax([scalar.value for scalar in self.scalars])
        for i in range(len(self.scalars)):
            self.scalars[i].value = softmax_values[i]

    def calculate_cost(self, target: Vector) -> Scalar:
        """
        Calculates cross entropy cost between this vector and target
        :param target: vector
        :return: cost between this and target
        """
        return CrossEntropyLoss(self.scalars, target.floats)
