import json
import logging
import os
import math
import random
from typing import Tuple
import time
from datetime import datetime as dt
from adam_optimizer import AdamOptimizer
from gradients import Gradients, Scalar, Vector

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

    def clear_gradients(self):
        """
        Clears gradients that were populated after a back propagation run
        """
        self.weights.clear_gradients()
        self.bias.visited = False
        self.bias.gradient = 0.0

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

    @property
    def weights(self) -> list[Vector]:
        return [n.weights for n in self.neurons]

    @property
    def biases(self) -> list[Scalar]:
        return [n.bias for n in self.neurons]

    def clear_gradients(self):
        """
        Clears gradients that were populated after a back propagation run
        """
        for neuron in self.neurons:
            neuron.clear_gradients()

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

    def clear_gradients(self):
        """
        Clears gradients that were populated after a back propagation run
        """
        for layer in self.layers:
            layer.clear_gradients()

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
        self.weight_optimizer = AdamOptimizer()
        self.bias_optimizer = AdamOptimizer()
        self.progress = []
        self.training_data_buffer: list[Tuple[Vector, Vector]] = []
        self.training_buffer_size = self._calculate_buffer_size(layer_sizes)
        self.training_sample_size = int(self.training_buffer_size * 0.1) # sample 10% of buffer

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

    @property
    def weights(self) -> list[list[Vector]]:
        return [l.weights for l in self.layers]

    @property
    def biases(self) -> list[list[Scalar]]:
        return [l.biases for l in self.layers]
    
    @property
    def activation_algos(self) -> list[str]:
        return [l.activation_algo for l in self.layers]
    
    def get_model_data(self) -> dict:
        return {
            "weights": [[[w.value for w in wv.scalars] for wv in lwv] for lwv in self.weights],
            "weight_optimizer_state": self.weight_optimizer.state,
            "biases": [[b.value for b in lb] for lb in self.biases],
            "bias_optimizer_state": self.bias_optimizer.state,
            "activation_algos": self.activation_algos,
            "progress": self.progress,
            "training_data_buffer": [tuple(tv.floats for tv in ttv) for ttv in self.training_data_buffer],
        }

    @weights.setter
    def weights(self, new_weights: list[list[list[float]]]):
        for layer, new_layer_weights in zip(self.layers, new_weights):
            for neuron, new_neuron_weights in zip(layer.neurons, new_layer_weights):
                neuron.weights = Vector(new_neuron_weights)

    @biases.setter
    def biases(self, new_biases: list[list[float]]):
        for layer, new_layer_biases in zip(self.layers, new_biases):
            for neuron, new_neuron_bias in zip(layer.neurons, new_layer_biases):
                neuron.bias = Scalar(new_neuron_bias)

    @activation_algos.setter
    def activation_algos(self, new_algos: list[str]):
        for layer, new_algo in zip(self.layers, new_algos):
            layer.activation_algo = new_algo
            for neuron in layer.neurons:
                neuron.activation_algo = new_algo

    def set_model_data(self, model_data: dict):
        self.weights = model_data["weights"]
        self.weight_optimizer.state = model_data["weight_optimizer_state"]
        self.biases = model_data["biases"]
        self.bias_optimizer.state = model_data["bias_optimizer_state"]
        self.activation_algos = model_data["activation_algos"]
        self.progress = model_data["progress"]
        self.training_data_buffer = [tuple(Vector(t) for t in tt) for tt in model_data["training_data_buffer"]]

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
        layer_sizes = [len(model_data["weights"][0])] + [len(w[0]) for w in model_data["weights"]]
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

    def compute_output(self, input_vector: Vector, target_vector: Vector = None, dropout_rate=0.0) -> Tuple[Vector, float, Gradients]:
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
            algo = self.activation_algos[l]
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
            cost = cost_scalar.value
            # clear gradients
            self.clear_gradients()
            # back propagate to populate gradients
            cost_scalar.back_propagate()
        # populate gradients
        gradients = Gradients(self.weights, self.biases)

        return activation, cost, gradients

    def _train_step(self, avg_gradients: Gradients, learning_rate: float, l2_lambda: float):
        """
        Update the weights and biases of the neural network using the averaged cost derivatives.
        :param avg_gradients: Averaged gradients with respect to weights and biases
        :param learning_rate: Learning rate for gradient descent.
        :param l2_lambda: L2 regularization strength.
        """
        # Optimize weight gradients
        optimized_weight_steps = self.weight_optimizer.step(avg_gradients.wrt_weights, learning_rate)
        # Update weights by optimized gradients
        for layer_weights, optimized_weight_step in zip(self.weights, optimized_weight_steps):
            for weight_vector, optimized_weight_gradients in zip(layer_weights, optimized_weight_step):
                for weight_scalar, optimized_weight_gradient in zip(weight_vector.scalars, optimized_weight_gradients):
                    weight_scalar.value -= optimized_weight_gradient
        # Update weights with L2 regularization
        for layer_weights in self.weights:
            for weight_vector in layer_weights:
                for weight_scalar in weight_vector.scalars:
                    l2_penalty = l2_lambda * weight_scalar.value
                    weight_scalar.value -= l2_penalty
        # Optimize bias gradients
        optimized_bias_steps = self.bias_optimizer.step(avg_gradients.wrt_biases, learning_rate)
        # Update biases
        for layer_biases, optimized_bias_step in zip(self.biases, optimized_bias_steps):
            for bias_scalar, optimized_bias_gradient in zip(layer_biases, optimized_bias_step):
                bias_scalar.value -= optimized_bias_gradient

    def train(self, training_data: list[Tuple[Vector, Vector]], epochs=100, learning_rate=0.01, decay_rate=0.9,
              dropout_rate=0.2, l2_lambda=0.001):
        """
        Train the neural network using the provided training data.
        :param training_data: List of tuples [(input_vector, target_vector), ...].
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers
        :param l2_lambda: L2 regularization strength
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

            # Calculate total cost and average gradients
            avg_gradients = Gradients(self.weights, self.biases)
            total_cost = 0
            for i, (input_vector, target_vector) in enumerate(training_data_sample):
                _, cost, gradients = self.compute_output(input_vector, target_vector, dropout_rate)
                total_cost += cost
                avg_gradients.take_avg(gradients, 1.0 / (i + 2))

            # Update weights and biases
            current_learning_rate = learning_rate * (decay_rate ** epoch)
            self._train_step(avg_gradients, current_learning_rate, l2_lambda)

            # Record progress
            self.progress.append({
                "dt": dt.now().isoformat(),
                "epoch": epoch + 1,
                "cost": total_cost / len(training_data_sample)
            })
            last_progress = self.progress[-1]
            print(f"Model {self.model_id}: {last_progress["dt"]} - Epoch {last_progress["epoch"]}, "
                  f"Cost: {last_progress["cost"]:.4f} ")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Serialize model after training
        self.serialize()
