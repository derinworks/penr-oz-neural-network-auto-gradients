import math
import random

def sigmoid(x: float) -> float:
    """
    Takes an input x and returns a result of the same shape, avoiding overflow errors.
    :param x: an input
    :return: Sigmoid activation of x
    """
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    """
    Takes an input x and returns the derivative of sigmoid activation.
    :param x: an input
    :return: Derivative of sigmoid activation
    """
    sig = sigmoid(x) if x < 0 or x > 1 else x
    return sig * (1 - sig)

def relu(x: float) -> float:
    """
    Takes an input x and applies the ReLU activation function.
    :param x: an input
    :return: ReLU activation of x
    """
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    """
    Takes an input x and returns the derivative of the ReLU function.
    :param x: an input
    :return: Derivative of ReLU activation
    """
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    """
    Takes an input x and applies the tanh activation function.
    :param x: an input
    :return: tanh activation of x
    """
    # Check for invalid values
    if math.isnan(x) or math.isinf(x):
        raise ValueError("Input contains NaN or Inf values.")
    return math.tanh(x)

def tanh_derivative(x: float) -> float:
    """
    Takes an input x and returns the derivative of the tanh function.
    :param x: an input
    :return: Derivative of tanh activation
    """
    return 1 - x ** 2

def mean_squared_error(x, y):
    """
    Compares x and y and calculates cost going from x to y
    :param x: first
    :param y: second
    :return: the mean squared error between x to y
    """
    return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])

def softmax(x: list[float]) -> list[float]:
    """
    Compute the softmax of a vector.
    :param x: a vector
    :return: Softmax probabilities of the input.
    """
    # Check for invalid values
    if any(math.isnan(xi) or math.isinf(xi) for xi in x):
        raise ValueError("Input contains NaN or Inf values.")
    # Shift input values for numerical stability (prevent overflow/underflow)
    max_x = max(x)
    shift_x = [xi - max_x for xi in x]
    # Exponential of shifted values
    exp_x = [math.exp(xi) for xi in shift_x]
    # Normalize by the sum of exponential
    sum_exp_x = sum(exp_x)
    return [exp_xi / sum_exp_x for exp_xi in exp_x]

def cross_entropy_gradient(x: float, y: float) -> float:
    """
    Compute the gradient of the cross-entropy loss with softmax for a single sample.
    :param x: an input (pre-activation logit for a single sample).
    :param y: an expected output (a probability value).
    :return: Gradient of the loss with respect to the logit.
    """
    return x - y

def cross_entropy_loss(x: list[float], y: list[float]) -> float:
    """
    Compute the cross-entropy loss for a single sample.
    :param x: an input (1D pre-activation logits for a single sample).
    :param y: an expected output (1D expected output, either one-hot encoded or a probability distribution).
    :return: Cross entropy loss.
    """
    # Clipping to avoid log(0) issues
    eps = 1e-12
    x = [max(min(p, 1 - eps), eps) for p in x]
    # Compute cross-entropy loss
    return -sum(math.log(xi) * yi for xi, yi in zip(x, y))

def batch_norm(x: list[float], epsilon=1e-5):
    """
    Applies batch normalization to given input
    :param x: an input
    :param epsilon: normalization option
    :return: batch normalized result
    """
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    scale = 1.0 / math.sqrt(variance + epsilon)
    return [(xi - mean) * scale for xi in x]

def apply_dropout(x: list[float], dropout_rate: float):
    """
    Applies dropout to the given input.
    :param x: an input
    :param dropout_rate: The fraction of entries to drop (e.g., 0.2 for 20%).
    :return: result with dropout applied.
    """
    if dropout_rate <= 0.0 or dropout_rate >= 1.0:
        return x  # No dropout if rate is invalid
    scale = 1.0 / (1.0 - dropout_rate)
    return [xi * scale if random.random() > dropout_rate else 0 for xi in x]
