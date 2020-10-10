from typing import List, Dict, Any
import numpy as np
from SOM.Neuron import Neuron

class NeuralNetwork:
    def __init__(self,
                 n_layers: int,
                 neurons: List[Neuron] = list()):
        self.__neurons = neurons
        self.__synapse: Dict[np.ndarray, Neuron] = dict()
        self.__n_layers: int = n_layers

    def fit(self, data_training: np.ndarray, learning_rate: float):
        pass

    def set_layer(self, layer_number: int = 1):
        pass