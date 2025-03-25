import json
import logging
import os
import math
import random
from typing import Tuple
import time
from datetime import datetime as dt
from gradients import Scalar, Vector

log = logging.getLogger(__name__)

class Neuron:
    def __init__(self, input_size: int, weight_algo="xavier", bias_algo="random", activation_algo="relu"):
        """
        Initialize a neuron
        :param input_size: represents the input size of neuron
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for bias (default: "random")
        :param activation_algo: Activation algorithm (default: "relu")
        """
        self.weights = Vector([
            Scalar(
                random.uniform(-math.sqrt(1 / input_size), math.sqrt(1 / input_size)) if weight_algo == "xavier"
                else random.uniform(-math.sqrt(2 / input_size), math.sqrt(2 / input_size)) if weight_algo == "he"
                else random.uniform(-1, 1) # gaussian
            )
            for _ in range(input_size)
        ])
        self.bias = Scalar(0) if bias_algo == "zeros" else Scalar(random.uniform(-1, 1))
        self.activation_algo = activation_algo

    def output(self, input_vector: Vector) -> Scalar:
        """
        Gives output of this neuron pre-activation for given input.
        :param input_vector: an input vector
        :return: output scalar
        """
        return self.weights.dot(input_vector) + self.bias

    def activate(self, input_vector: Vector) -> Scalar:
        """
        Activates this neuron with given input.
        :param input_vector: an input vector
        :return: activated output scalar
        """
        return self.output(input_vector).activate(self.activation_algo)

class Layer:
    def __init__(self, input_size: int, output_size: int, weight_algo="xavier", bias_algo="random", activation_algo="relu"):
        """
        Initialize a layer of neurons
        :param input_size: represents the input size of layer
        :param output_size: represents the output size of layer
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for bias (default: "random")
        :param activation_algo: Activation algorithm (default: "relu")
        """
        self.neurons = [Neuron(input_size, weight_algo, bias_algo, activation_algo)
                        for _ in range(output_size)]
        self.activation_algo = activation_algo

    def output(self, input_vector: Vector) -> Vector:
        """
        Gives output of this layer of neurons pre-activation for given input
        :param input_vector: an input vector
        :return: output vector pre-activation
        """
        return Vector([neuron.output(input_vector) for neuron in self.neurons])
    
    def activate(self, input_vector: Vector) -> Vector:
        """
        Activates this layer of neurons with given input.
        :param input_vector: an input vector
        :return: activated output vector
        """
        return self.output(input_vector).activate(self.activation_algo)

class MultiLayerPerceptron:
    def __init__(self, layer_sizes: list[int],
                 weight_algo: str = "xavier",
                 bias_algo: str = "random",
                 activation_algos: list[str] = None):
        """
        Initialize a multi-layer perceptron
        :param layer_sizes: List of integers where each integer represents the input size of the corresponding layer
        and the output size of the next layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms per layer (default: "relu")
        """
        input_sizes = layer_sizes[:-1]
        output_sizes = layer_sizes[1:]
        if activation_algos is None:
            activation_algos = ["relu"] * len(input_sizes)
        self.layers = [Layer(input_size, output_size, weight_algo, bias_algo, activation_algo)
                       for input_size, output_size, activation_algo in
                       zip(input_sizes, output_sizes, activation_algos)]

    def activate(self, input_vector: Vector) -> Vector:
        """
        Activates this multi-layer perceptron with given input.
        :param input_vector: an input vector
        :return: activated output vector
        """
        # no layers means output same as input
        output_vector = input_vector
        for layer in self.layers:
            # feed forward output of each layer
            output_vector = layer.activate(output_vector)
        # output of final layer
        return output_vector

class NeuralNetworkModel(MultiLayerPerceptron):
    def __init__(self, model_id, layer_sizes: list[int], weight_algo="xavier", bias_algo="random", activation_algos=None):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms (default: "sigmoid")
        """
        super().__init__(layer_sizes, weight_algo, bias_algo, activation_algos)
        self.model_id = model_id
        self.progress = []
        self.training_data_buffer: list[Tuple[Vector, Vector]] = []
        self.training_buffer_size = self._calculate_buffer_size(layer_sizes)
        self.training_sample_size = int(self.training_buffer_size * 0.1) # sample 10% of buffer
        self.avg_cost = None

    @staticmethod
    def _calculate_buffer_size(layer_sizes: list[int]) -> int:
        """
        Calculate training data buffer size based on total number of parameters in the network.
        """
        total_params = sum(
            layer_size * next_layer_size + next_layer_size
            for layer_size, next_layer_size in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        return total_params  # Buffer size is equal to total parameters

    @classmethod
    def _get_scalar_state(cls, scalar: Scalar) -> dict:
        return {
            "val": scalar.value,
            "t": scalar.adam_optimizer.t,
            "m": scalar.adam_optimizer.m,
            "v": scalar.adam_optimizer.v
        }

    @classmethod
    def _get_scalar(cls, state: dict) -> Scalar:
        scalar = Scalar(state["val"])
        scalar.adam_optimizer.t = state["t"]
        scalar.adam_optimizer.m = state["m"]
        scalar.adam_optimizer.v = state["v"]
        return scalar

    def get_model_data(self) -> dict:
        return {
            "layers": [{
                "algo": l.activation_algo,
                "neurons": [{
                    "weights": [self._get_scalar_state(w) for w in n.weights.scalars],
                    "bias": self._get_scalar_state(n.bias)
                } for n in l.neurons]
            } for l in self.layers],
            "progress": self.progress,
            "training_data_buffer": [tuple(tv.floats for tv in ttv) for ttv in self.training_data_buffer],
            "average_cost": self.avg_cost,
        }

    def set_model_data(self, model_data: dict):
        for layer, layer_state in zip(self.layers, model_data["layers"]):
            layer.activation_algo = layer_state["algo"]
            for neuron, neuron_state in zip(layer.neurons, layer_state["neurons"]):
                neuron.activation_algo = layer_state["algo"]
                neuron.weights = Vector([self._get_scalar(w) for w in neuron_state["weights"]])
                neuron.bias = self._get_scalar(neuron_state["bias"])

        self.progress = model_data["progress"]
        self.training_data_buffer = [tuple(Vector(t) for t in tt) for tt in model_data["training_data_buffer"]]
        self.avg_cost = model_data["average_cost"]

    def serialize(self):
        filepath = f"model_{self.model_id}.json"
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = self.get_model_data()
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        log.info(f"Model saved successfully: {full_path}")

    @classmethod
    def deserialize(cls, model_id: str):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except FileNotFoundError as e:
            log.error(f"File not found error occurred: {str(e)}")
            raise KeyError(f"Model {model_id} not created yet.")
        layer_state = model_data["layers"]
        input_size = len(layer_state[0]["neurons"][0]["weights"])
        layer_sizes = [input_size] + [len(l["neurons"]) for l in layer_state]
        model = cls(model_id, layer_sizes)
        model.set_model_data(model_data)
        return model

    @classmethod
    def delete(cls, model_id: str):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        try:
            os.remove(full_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def compute_output(self, input_vector: Vector, target_vector: Vector = None, dropout_rate=0.0) -> Tuple[Vector, Scalar | None]:
        """
        Compute activated output and optionally also cost compared to the provided target vector.
        :param input_vector: Input vector
        :param target_vector: Target vector (optional)
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers (optional)
        :return: activation vector, cost, gradients
        """
        activation = input_vector
        num_layers = len(self.layers)
        for l in range(num_layers):
            algo = self.layers[l].activation_algo
            pre_activation: Vector = self.layers[l].output(activation)
            if algo == "relu" and l < num_layers - 1:
                # stabilize output in hidden layers prevent overflow with ReLU activations
                pre_activation.batch_norm()
            activation = pre_activation.activate(algo)
            if l < num_layers - 1:  # Hidden layers only
                # Apply dropout only to hidden layers
                activation.apply_dropout(dropout_rate)

        cost = None
        if target_vector is not None:
            cost_scalar: Scalar = activation.calculate_cost(target_vector)
            cost = cost_scalar

        return activation, cost

    def train(self, training_data: list[Tuple[Vector, Vector]], epochs=100, learning_rate=0.01, decay_rate=0.9,
              dropout_rate=0.2, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Train the neural network using the provided training data.
        :param training_data: List of tuples [(input_vector, target_vector), ...].
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers
        :param l2_lambda: L2 regularization strength
        :param beta1: adam optimizer first moment parameter
        :param beta2: adam optimizer second moment parameter
        :param epsilon: adam optimizer the smallest step parameter
        """
        # Combine incoming training data with buffered data
        self.training_data_buffer.extend(training_data)

        # Check if buffer size is sufficient
        if len(self.training_data_buffer) < self.training_buffer_size:
            print(f"Model {self.model_id}: Insufficient training data. "
                  f"Current buffer size: {len(self.training_data_buffer)}, "
                  f"required: {self.training_buffer_size}")
            self.serialize() # serialize model with partial training data for next time
            return

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        self.progress = []
        last_serialized = time.time()
        for epoch in range(epochs):
            random.shuffle(training_data)
            training_data_sample = training_data[:self.training_sample_size]

            # decay learning rate
            current_learning_rate = learning_rate * (decay_rate ** epoch)

            # Calculate cost and populate gradients
            avg_cost = None
            for i, (input_vector, target_vector) in enumerate(training_data_sample):
                _, cost = self.compute_output(input_vector, target_vector, dropout_rate)
                # calculate contribution factor for averaging
                alpha = 1.0 / (i + 2)
                # back propagate to populate gradients
                cost.back_propagate(alpha, current_learning_rate, beta1, beta2, epsilon)
                # calculate average cost
                avg_cost = alpha * (avg_cost or cost.value) + (1 - alpha) * cost.value

            # apply gradient descent
            for l in self.layers:
                for n in l.neurons:
                    n.bias.value -= n.bias.gradient_optimized
                    for w in n.weights.scalars:
                        w.value -= w.gradient_optimized
                        # Update weight with L2 regularization
                        w.value *= (1.0 - l2_lambda)

            # Record progress
            self.progress.append({
                "dt": dt.now().isoformat(),
                "epoch": epoch + 1,
                "cost": avg_cost
            })
            last_progress = self.progress[-1]
            print(f"Model {self.model_id}: {last_progress["dt"]} - Epoch {last_progress["epoch"]}, "
                  f"Cost: {last_progress["cost"]:.4f} ")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Calculate current average progress cost
        avg_progress_cost = sum([progress["cost"] for progress in self.progress]) / len(self.progress)
        # Update overall average cost
        self.avg_cost = ((self.avg_cost or avg_progress_cost) + avg_progress_cost) / 2.0
        # Log training result
        training_dt = dt.now().isoformat()
        print(f"Model {self.model_id}: {training_dt} - Done training for {epochs} epochs, "
               f"Cost: {avg_progress_cost:.4f} Overall Cost: {self.avg_cost:.4f}")

        # Serialize model after training
        self.serialize()
