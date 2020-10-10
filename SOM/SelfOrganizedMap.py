import numpy as np
from typing import List, Tuple, Union, Any
from SOM.Neuron import Neuron
from SOM.NeuralNetwork import NeuralNetwork
import random
from math import exp


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    d: np.ndarray = (a - b)**2
    d = d.sum()
    d: float = np.sqrt(d)
    return d


class SelfOrganizedMap(NeuralNetwork):
    def __init__(self,
                 neuron_per_dimension: Union[List, Tuple],
                 initial_radius: float
                 ):
        """
        Self Organized Map Class.

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

    def neighbors(self, neuron: Neuron) -> List[Neuron]:
        neighbors = []
        for line in self.__layer:
            for other_neuron in line:
                d = euclidean_dist(neuron.w, other_neuron.w)
                if d < self.__r:
                    neighbors.append(other_neuron)
        return neighbors

    def neuron_winner(self,
                      input_vector: np.ndarray,
                      dist_function: Any) -> Neuron:
        best_dist: float = float('inf')
        winner: Neuron = self.__layer[0][0]

        for line in self.__layer:
            for neuron in line:
                d = dist_function(input_vector, neuron.w)
                if d < best_dist:
                    best_dist = d
                    winner: Neuron = neuron
        return winner

    def fit(self,
            data_training: List[np.ndarray],
            activation_function: Any,
            dist_function: Any = euclidean_dist,
            learning_rate: float = .5) -> None:
        """
        Self organized Map Function
        :param data_training:
        :param activation_function:
        :param dist_function:
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
            winner = self.neuron_winner(x, dist_function=dist_function)

            # update neighbors
            neighbors = self.neighbors(winner)
            for neuron in neighbors:
                new_w = neuron.w + learning_rate * exp(- dist_function(winner.w, neuron.w)) / self.__r * (x - neuron.w)
                neuron.update_weight(new_w)

        self.__layer = layer

    def transform(self, input_vector: np.ndarray) -> np.ndarray:
        neurons = []
        positions = []
        for i, line in enumerate(self.__layer):
            for j, neuron in enumerate(line):
                if neuron.out(input_vector):
                    neurons.append(neuron)
                    positions.append([i, j])

        output = np.array(positions[0])
        return output
