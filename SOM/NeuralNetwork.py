from typing import List, Dict, Any
import numpy as np
from SOM.Neuron import Neuron

class NeuralNetwork:
    def __init__(self,
                 n_layers: int,
                 format_layer: List[int]):
        self.__n_layers: int = n_layers
        self.__format_layer: list = format_layer

    @property
    def format_layer(self):
        return self.__format_layer

    @property
    def n_layer(self):
        return self.__n_layers


    def fit(self, data_training: np.ndarray, learning_rate: float):
        pass
