import numpy as np
from typing import List
from SOM.Neuron import Neuron
from SOM.NeuralNetwork import NeuralNetwork
import random


class SelfOrganizedMap(NeuralNetwork):
    def __init__(self,
                 data_training: List[Neuron],
                 dimension_number: int,
                 neuron_per_dimension: int,
                 initial_radius: float
                 ):
        """

        
        :param data_training:
        :param dimension_number:
        :param neuron_per_dimension:
        :param initial_radius:
        """
        NeuralNetwork.__init__(n_layers=1,
                               neurons=list())

        self.__data_training: List[Neuron] = data_training
        self.__d: int = dimension_number
        self.__n: int = neuron_per_dimension
        self.__r: float = initial_radius

    @property
    def d(self):
        return self.__d

    @property
    def n(self):
        return self.__n

    @property
    def r(self):
        return self.__r

    def fit(self, data_training: list):
        pass

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        pass

    def neighbors(self, neuron: Neuron, radius: float) -> List[Neuron]:
        pass



    def som(self, learning_rate: float = .5):
        """
        Self organized Map Function
        :param learning_rate:
        :return:
        """

        radius = np.zeros((self.__d, self.__n))
        for dimension in range(self.__d):
            for neuron in range(self.__n):
                radius[dimension][neuron] = random.random()

        for neuron in self.__data_training:
            best_neuron = np.argmin([self.dist(a=neuron, b=radio) for radio in radius])
            for radio in radius:
                radio = radio + learning_rate * np.exp(-self.dist(best_neuron, radio)) / self.__r * (best_neuron - radio)

        return radius
