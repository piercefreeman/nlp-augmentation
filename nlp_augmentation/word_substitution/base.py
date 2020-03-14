from abc import ABC, abstractmethod
import numpy as np
from typing import List


class WordSubstitutionBase(ABC):
    def fit(self, examples: List[str]):
        """
        Handle any training on the full corpus that's necessary to initialize this word
        substitution model.  No-Op unless over-ridden by subclasses.

        :param example: list containing all corpus examples.

        """
        pass

    @abstractmethod
    def __call__(self, example: str):
        """
        Apply the word substitution on one example.

        :param example: str containing the desired text to augment.
        """
        pass

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token
