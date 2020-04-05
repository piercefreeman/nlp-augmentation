from nlp_augmentation.word_substitution.base import WordSubstitutionBase
from nlp_augmentation.base import AugmentationBase
import numpy as np
from nlp_augmentation.data_models import AugmentedDatapoint
from tqdm import tqdm


class UniformWordSubstitution(AugmentationBase, WordSubstitutionBase):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, augmentations, token_prob, vocab):
        super().__init__(augmentations=augmentations)

        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def fit(self, examples):
        pass

    def __call__(self, examples):
        return [
            [
                AugmentedDatapoint(
                    identifier=example.identifier,
                    augmented_index=augmented_index,
                    text=" ".join(self.replace_tokens(example.text.split()))
                )
                for augmented_index in range(self.augmentations)
            ]
            for example in tqdm(examples)
        ]

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            if np.random.random() < 0.001:
                show_example = True
            else:
                show_example = False
            if show_example:
                tf.logging.info("before augment: {:s}".format(
                        filter_unicode(" ".join(tokens))))
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()
            if show_example:
                tf.logging.info("after augment: {:s}".format(
                        filter_unicode(" ".join(tokens))))
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = self.vocab.keys()
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)

