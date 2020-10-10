import numpy as np
from typing import Any

class Neuron:
    def __init__(self,
                 weights: np.ndarray,
                 activation_function: Any,
                 threshold: float = 0
                 ):
        self.__threshold = threshold
        self.__weights = weights
        self.activation_function = activation_function

    @property
    def w(self):
        return self.__weights

    def update_weight(self, weights: np.ndarray):
        self.__weights = weights

    def out(self, input_vector: np.ndarray) -> bool:
        pass

    def neuron_func(self):
        pass
