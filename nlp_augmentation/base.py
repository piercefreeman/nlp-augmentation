from abc import ABC, abstractmethod
from typing import List


class AugmentationBase(ABC):
    def fit(self, examples: List[str]):
        """
        Handle any training on the full corpus that's necessary to initialize this word
        substitution model.  No-Op unless over-ridden by subclasses.

        :param example: list containing all corpus examples.

        """
        pass

    @abstractmethod
    def __call__(self, examples: List[str]):
        """
        Apply the word substitution on a batch of examples.

        :param example: str containing the desired text to augment.
        """
        pass
