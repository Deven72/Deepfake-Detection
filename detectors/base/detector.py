from abc import ABC, abstractmethod


class BaseDetector(ABC):

    @abstractmethod
    def predict(self, input_data):
        """
        Run detection on the given input.

        Returns:
            dict: Detection result
        """
        pass