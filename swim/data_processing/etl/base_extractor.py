from abc import ABC, abstractmethod

class BaseDataExtractor(ABC):
    @abstractmethod
    def extract(self):
        """
        Extract data from the source.
        """
        pass