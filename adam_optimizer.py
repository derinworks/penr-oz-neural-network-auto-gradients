import math

class AdamOptimizer:
    def __init__(self):
        self.t: int = 0
        self.m = 0.0
        self.v = 0.0

    def step(self, gradient: float, learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) -> float:
        """
        Perform a single Adam optimization step.
        :param gradient: a float representing gradient
        :param learning_rate: Learning rate.
        :param beta1: parameter for first moment (m: mean)
        :param beta2: parameter for second moment (v: variance)
        :param epsilon: parameter for the smallest step
        :return: optimized gradient
        """
        # Increment time step
        self.t += 1
        # Update biased moment estimates
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * (gradient ** 2)
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        # Compute bias-corrected learning rate
        alpha = learning_rate * math.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t)
        # Compute step
        step = alpha * m_hat / (math.sqrt(v_hat) + epsilon)
        # return optimized gradient
        return step