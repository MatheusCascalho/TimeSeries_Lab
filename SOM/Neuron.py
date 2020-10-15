import numpy as np
from typing import Any


class ActivationFunction:
    def __init__(self,
                 function: Any,
                 threshold: float = None):
        self.__function = function
        self.__threshold = threshold

    @property
    def threshold(self):
        return self.__threshold

    def transform(self, input_vector: np.ndarray):
        if self.__threshold is not None:
            return self.__function(input_vector, threshold=self.threshold)
        else:
            return self.__function(input_vector)


class Neuron:
    def __init__(self,
                 weights: np.ndarray,
                 activation_function: Any,
                 ):
        self.__weights = weights
        self.activation_function = activation_function

    def __repr__(self):
        return 'Neuron'

    @property
    def w(self):
        return self.__weights

    def update_weight(self, weights: np.ndarray):
        self.__weights = weights

    def out(self, input_vector: np.ndarray) -> float:
        d = self.__function(input_vector, self.__weights)
        return d

def sigmoid(x: list):
    pass

def step_function(x: list):
    pass

# Neuron(activation_function=sigmoid)