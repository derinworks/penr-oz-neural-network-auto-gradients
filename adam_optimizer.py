import math

class AdamOptimizer:
    def __init__(self):
        self.t: int = 0
        self.m = 0.0
        self.v = 0.0

    @property
    def _beta1(self) -> float:
        return 0.95 - 0.1 * math.exp(-1e-4 * self.t)

    @property
    def _beta2(self) -> float:
        return 0.99 - 0.1 * (self.t / (self.t + 1000.0))

    @property
    def _epsilon(self) -> float:
        return 1e-8 + 0.01 * (1 - math.exp(-1e-6 * self.t))

    def step(self, gradient: float, learning_rate=0.1) -> float:
        """
        Perform a single Adam optimization step.
        :param gradient: a float representing gradient
        :param learning_rate: Learning rate.
        :return: optimized gradient
        """
        # Increment time step
        self.t += 1
        # Update biased moment estimates
        self.m = self._beta1 * self.m + (1 - self._beta1) * gradient
        self.v = self._beta2 * self.v + (1 - self._beta2) * (gradient ** 2)
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self._beta1 ** self.t)
        v_hat = self.v / (1 - self._beta2 ** self.t)
        # Compute step
        step = learning_rate * m_hat / (math.sqrt(v_hat) + self._epsilon)
        # return optimized gradient
        return step