"""Fixed inference adapters built only from validated numerical parameters.

These classes never import a class named by an artifact. Training state and
optimizer objects are deliberately absent. Arrays are owned immutable buffers.
"""
import numpy as np


def owned_array(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class AffinePredictor:
    """Linear/PLS affine prediction with explicit historical normalization."""
    def __init__(self, coefficients, intercept, *, center=None, scale=None, kind="linear"):
        self.coefficients = owned_array(coefficients)
        self.intercept = owned_array(intercept)
        self.center = None if center is None else owned_array(center)
        self.scale = None if scale is None else owned_array(scale)
        self.kind = kind

    def predict(self, X):
        values = np.asarray(X)
        if self.center is not None:
            values = values.copy()
            values -= self.center
        if self.scale is not None:
            values /= self.scale
        return values @ self.coefficients.T + self.intercept


class MLPPredictor:
    """The sklearn dense MLP forward, with known activation functions only."""
    def __init__(self, weights, biases, *, activation, output_activation):
        self.weights = tuple(owned_array(value) for value in weights)
        self.biases = tuple(owned_array(value) for value in biases)
        self.activation = activation
        self.output_activation = output_activation

    @staticmethod
    def _activate(values, name):
        if name == "identity":
            return values
        if name == "relu":
            np.maximum(values, 0, out=values)
        elif name == "tanh":
            np.tanh(values, out=values)
        elif name == "logistic":
            from scipy.special import expit
            expit(values, out=values)
        elif name == "exponential":
            np.exp(values, out=values)
        else:  # Constructor is called only after artifact schema validation.
            raise ValueError("unsupported portable MLP activation")
        return values

    def predict(self, X):
        values = np.asarray(X)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight
            values += bias
            name = self.output_activation if index == len(self.weights) - 1 else self.activation
            self._activate(values, name)
        return values


class TorchPredictor:
    """CPU float32 eval graph of whitelisted package NN layer operations.

    Functional operators avoid layer initialization, RNG use, torch.load and
    executable graph deserialization. Dropout is an explicit eval-time identity.
    """
    def __init__(self, layers):
        self.layers = tuple({key: owned_array(np.asarray(value, dtype=np.float32)) if isinstance(value, np.ndarray) else value
                             for key, value in layer.items()} for layer in layers)

    def predict(self, X):
        import torch
        import torch.nn.functional as functional
        values = torch.tensor(np.asarray(X), dtype=torch.float32, device="cpu")
        with torch.no_grad():
            for layer in self.layers:
                kind = layer["type"]
                if kind == "linear":
                    weight = torch.tensor(layer["weight"], dtype=torch.float32, device="cpu")
                    bias = None if layer["bias"] is None else torch.tensor(layer["bias"], dtype=torch.float32, device="cpu")
                    values = functional.linear(values, weight, bias)
                elif kind == "batchnorm":
                    def tensor(name):
                        return None if layer[name] is None else torch.tensor(layer[name], dtype=torch.float32, device="cpu")
                    values = functional.batch_norm(values, tensor("running_mean"), tensor("running_var"),
                                                   tensor("weight"), tensor("bias"), training=False,
                                                   momentum=0.0, eps=layer["eps"])
                elif kind == "relu":
                    values = functional.relu(values)
                elif kind == "tanh":
                    values = torch.tanh(values)
                elif kind == "sigmoid":
                    values = torch.sigmoid(values)
                elif kind not in ("dropout", "identity"):
                    raise ValueError("unsupported portable Torch layer")
        return values.numpy()
