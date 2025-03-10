import functions as func
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
        self.gradient: float | None = None
        self.visited = False

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
        if self.gradient is None:
            self.gradient = 1.0

    def back_propagate(self, target=None):
        """
        Applies back propagation to compute gradient of this and the previous operand scalars
        :param target: not used at scalar level back propagation
        """
        if not self.visited:
            self.visited = True
            self._compute_gradient()
            for operand in self.operands:
                operand.back_propagate()

class Summation(Scalar):
    def __init__(self, left_addend: Scalar, right_addend: Scalar):
        """
        Initialize a summation result scalar with left and right addend scalars
        :param left_addend: left addend scalar
        :param right_addend: right addend scalar
        """
        super().__init__(left_addend.value + right_addend.value, (left_addend, right_addend))

    def _compute_gradient(self):
        super()._compute_gradient()
        for addend in self.operands:
            if addend.gradient is None:
                addend.gradient = self.gradient

class Multiplication(Scalar):
    def __init__(self, multiplicand: Scalar, multiplier: Scalar):
        """
        Initialize a multiplication result scalar with multiplicand and multiplier scalars
        :param multiplicand: scalar being multiplied
        :param multiplier: scalar multiplying
        """
        super().__init__(multiplicand.value * multiplier.value, (multiplicand, multiplier))

    def _compute_gradient(self):
        super()._compute_gradient()
        multiplicand, multiplier = self.operands
        if multiplicand.gradient is None:
            multiplicand.gradient = multiplier.value * self.gradient
        if multiplier.gradient is None:
            multiplier.gradient = multiplicand.value * self.gradient

class Exponentiation(Scalar):
    def __init__(self, base: Scalar, exponent: Scalar):
        """
        Initialize an exponentiation result scalar with base and exponent scalars
        :param base: base scalar
        :param exponent: exponent scalar
        """
        super().__init__(base.value**exponent.value, (base, exponent))

    def _compute_gradient(self):
        super()._compute_gradient()
        base, exponent = self.operands
        if base.gradient is None:
            base.gradient = (exponent.value * base.value**(exponent.value - 1)) * self.gradient

class SigmoidActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a sigmoid activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.sigmoid(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        if self.gradient is None:
            self.gradient = func.sigmoid_derivative(self.value)

class ReluActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a ReLu activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.relu(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        if self.gradient is None:
            self.gradient = func.relu_derivative(self.value)

class TanhActivation(Scalar):
    def __init__(self, pre_activation: Scalar):
        """
        Initialize a tanh activation result scalar from pre-activated scalar
        :param pre_activation: pre-activated scalar
        """
        super().__init__(func.tanh(pre_activation.value), (pre_activation, ))

    def _compute_gradient(self):
        if self.gradient is None:
            self.gradient = func.tanh_derivative(self.value)

class Vector:
    def __init__(self, values: list[Scalar | float]):
        """
        Initialize a vector with values
        :param values: scalars or floats populating the vector
        """
        self.values = values

    def __repr__(self):
        return f"{type(self).__name__}({self.values})"

    @property
    def floats(self) -> list[float]:
        return [v.value if isinstance(v, Scalar) else v for v in self.values]

    @property
    def scalars(self) -> list[Scalar]:
        return [v if isinstance(v, Scalar) else Scalar(v) for v in self.values]

    def clear_gradients(self):
        """
        Clears gradients that were populated after a back propagation run
        """
        for scalar in self.scalars:
            scalar.gradient = None
            scalar.visited = False

    def dot(self, other):
        """
        Takes dot product of this vector with another one
        :param other: another vector
        :return: value of the dot product
        """
        return sum(v1 * v2 for v1, v2 in zip(self.values, other.values))

    def activate(self, algo: str):
        """
        Activates vector values
        :param algo: The activation algorithm ("softmax").
        :return: vector with activated values
        """
        if algo == "softmax":
            return SoftmaxActivation(self.scalars)
        else:
            self.values = [scalar.activate(algo) for scalar in self.scalars]
            return self

    def back_propagate(self, target = None):
        """
        Applies back propagation to all scalars
        :param target: target vector
        """
        for value in self.scalars:
            value.back_propagate()

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

    def back_propagate(self, target: Vector = None):
        """
        Applies softmax cross entropy back propagation to all scalars
        :param target: target vector to use for cross entropy
        """
        if target is not None:
            # compute gradients with softmax cross entropy first
            logits = [scalar.value for scalar in self.scalars]
            gradients = func.softmax_cross_entropy_gradient(logits, target.values)
            for i in range(len(self.scalars)):
                self.scalars[i].gradient = gradients[i]
        # then back propagate
        super().back_propagate(target)

class Gradients:
    def __init__(self, weights: list[list[Vector]], biases: list[list[Scalar]]):
        """
        Initialize gradients from given weights and biases
        :param weights: list of weights
        :param biases: list of biases
        """
        self.wrt_weights = [[[w.gradient or 0.0 for w in wv.scalars] for wv in lwv] for lwv in weights]
        self.wrt_biases = [[b.gradient or 0.0 for b in lb] for lb in biases]

    @classmethod
    def _take_avg(cls, gv: list[float], ogv: list[float], alpha: float):
        for i, (g, og) in enumerate(zip(gv, ogv)):
            gv[i] = alpha * g + (1 - alpha) * og

    def take_avg(self, other, alpha = 0.5):
        """
        Take average of these gradients with other gradients and update these
        :param other: other gradients
        :param alpha: influence of these gradients vs others during averaging
        """
        for lg, olg in zip(self.wrt_weights, other.wrt_weights):
            for gv, ogv in zip(lg, olg):
                self._take_avg(gv, ogv, alpha)

        for gv, ogv in zip(self.wrt_biases, other.wrt_biases):
            self._take_avg(gv, ogv, alpha)
