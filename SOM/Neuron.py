import numpy as np
from typing import Any

class Neuron:
    def __init__(self,
                 threshold,
                 weights: np.ndarray,
                 activation_function: Any):
        self.__threshold = threshold
        self.__weights = weights
        self.activation_function = activation_function

    def out(self, input_vector: np.ndarray) -> bool:
        pass

    def neuron_func(self):
        pass
