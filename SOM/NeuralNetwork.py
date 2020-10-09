from typing import List, Dict, Any
import numpy as np
from SOM.Neuron import Neuron

class NeuralNetwork:
    def __init__(self,
                 n_layers: int,
                 neurons: List[Neuron]):
        self.__neurons = neurons
        self.__synapse: Dict[np.ndarray, Neuron] = dict()

    def fit(self):
        pass
