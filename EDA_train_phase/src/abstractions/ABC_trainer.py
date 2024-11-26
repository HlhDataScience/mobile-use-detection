"""This module holds the abstract class for the train and test. It is thought to be instanciated and wotk seemlessly with the other classes defined in this project."""

from abc import ABC, abstractmethod


class BasicTrainer(ABC):
    """
    THis class is an abstraction to create custom trainer classes.
    This i the second part of the module and follows the instantiation of BasicPipeline class
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """This mandatory function handles the training of the model"""
        pass

    @abstractmethod
    def test(self):
        """THis mandatory method handles testing"""
        pass

    @abstractmethod
    def run(self):
        """This mandatory methods runs the full trainer"""
