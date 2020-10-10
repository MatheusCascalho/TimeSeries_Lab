import numpy as np
from typing import List, Tuple, Union, Any
from SOM.Neuron import Neuron
from SOM.NeuralNetwork import NeuralNetwork
import random
from math import exp

class SelfOrganizedMap(NeuralNetwork):
    def __init__(self,
                 neuron_per_dimension: Union[List, Tuple],
                 initial_radius: float
                 ):
        """

        
        :param data_training:
        :param dimension_number:
        :param neuron_per_dimension:
        :param initial_radius:
        """
        super().__init__(n_layers=1)
        self.__n: List = neuron_per_dimension
        self.__r: float = initial_radius
        self.__layer: np.ndarray = np.array([])

    @property
    def n(self) -> List[int]:
        """
        Neurons per dimension.
        :return:
        """
        return self.__n

    @property
    def r(self):
        return self.__r

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        dist: np.ndarray = (a - b)**2
        dist = dist.sum()
        dist: float = np.sqrt(dist)
        return dist

    def neighbors(self, neuron: Neuron) -> List[Neuron]:
        neighbors = []
        for line in self.__layer:
            for other_neuron in line:
                dist = self.dist(neuron.w, other_neuron.w)
                if dist < self.__r:
                    neighbors.append(other_neuron)
        return neighbors

    def neuron_winner(self,
                      input: np.ndarray) -> Neuron:
        best_dist: float = float('inf')
        winner: Neuron = self.__layer[0][0]

        for line in self.__layer:
            for neuron in line:
                dist = self.dist(input, neuron.w)
                if dist < best_dist:
                    best_dist = dist
                    winner: Neuron = neuron
        return winner

    def fit(self,
            data_training: List[np.ndarray],
            activation_function: Any,
            learning_rate: float = .5) -> None:
        """
        Self organized Map Function
        :param data_training:
        :param activation_function:
        :param learning_rate:
        :return:
        """
        n: int = len(self.__n)
        len_vector: int = len(data_training[0])
        lines = []

        # Initialization
        for i in range(n):
            d = self.__n[i]
            line = []
            for j in range(d):
                w = np.array([random.random() for _ in range(len_vector)])
                line.append(Neuron(weights=w, activation_function=activation_function))
            lines.append(line)

        layer: np.ndarray = np.array(lines)

        for x in data_training:
            winner = self.neuron_winner(x)

            # update neighbors
            neighbors = self.neighbors(winner)
            for neuron in neighbors:
                new_w = neuron.w + learning_rate*exp(-self.dist(winner.w, neuron.w)) / self.__r * (x - neuron.w)
                neuron.update_weight(new_w)

        self.__layer = layer
