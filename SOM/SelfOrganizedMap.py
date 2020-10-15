import numpy as np
from typing import List, Tuple, Union, Any
from SOM.Neuron import Neuron
from SOM.NeuralNetwork import NeuralNetwork
import random
from math import exp
from numpy import linalg as la


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    d: np.ndarray = (a - b)**2
    d = d.sum()
    d: float = np.sqrt(d)
    return d


class SelfOrganizedMap(NeuralNetwork):
    def __init__(self,
                 neuron_per_dimension: list,
                 initial_radius: float
                 ):
        """
        Self Organized Map Class.

        :param neuron_per_dimension:
        :param initial_radius:
        """
        super().__init__(n_layers=1, format_layer=neuron_per_dimension)
        self.__r: float = initial_radius
        self.__layer: np.ndarray = np.zeros(tuple(neuron_per_dimension))

    @property
    def r(self):
        return self.__r


    @property
    def layer(self):
        return self.__layer

    def neighbors(self, neuron: Neuron) -> List[Neuron]:
        neighbors = []
        for line in self.__layer:
            for other_neuron in line:
                d = la.norm(neuron.w) - la.norm(other_neuron.w)
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
                d = la.norm(input_vector) - la.norm(neuron.w)
                if d < best_dist:
                    best_dist = d
                    winner: Neuron = neuron
        return winner

    def fit(self,
            data_training: List[np.ndarray],
            activation_function: Any = euclidean_dist,
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
        n: int = len(self.format_layer)
        len_vector: int = len(data_training)
        shape: Tuple = data_training[0].shape
        lines = []

        # Initialization
        for i in range(n):
            d = self.format_layer[i]
            line = []
            for j in range(d):
                w = np.random.random_integers(low=0, high=255, size=shape)
                line.append(Neuron(weights=w, activation_function=activation_function))
            lines.append(line)

        self.__layer: np.ndarray = np.array(lines)

        for x in data_training:
            winner = self.neuron_winner(x, dist_function=dist_function)

            # update neighbors
            neighbors = self.neighbors(winner)
            for neuron in neighbors:
                new_w = neuron.w + learning_rate * exp(- dist_function(winner.w, neuron.w)) / self.__r * (x - neuron.w)
                neuron.update_weight(new_w)

        # self.__layer = layer

    def random_vector(self, shape: Tuple[int, int, int]):
        multidimensional = []
        for i in range(shape[0]):
            matrix = []
            for j in range(shape[1]):
                vector = [random.randint(0, 255) for _ in range(shape[2])]
                matrix.append(vector)
            multidimensional.append(matrix)
        return multidimensional

    def transform(self, input_vector: np.ndarray) -> tuple:
        result = (0, 0)
        best_distance = float('inf')
        for i, line in enumerate(self.__layer):
            for j, neuron in enumerate(line):
                dist = euclidean_dist(neuron.w, input_vector)
                if dist < best_distance:
                    best_distance = dist
                    result = (i, j)
        return result


if __name__=='__main__':
    import numpy as np
    from PIL import Image

    f = '../sky.jpeg'
    sky: Image = Image.open(f)
    data_image = np.asarray(sky.convert('RGB'))
    som = SelfOrganizedMap(neuron_per_dimension=[40, 40], initial_radius=10)
    som.fit(list([data_image]))
    # print(data_image[21])
    print(som.transform(data_image))
    # print(la.norm(data_image))