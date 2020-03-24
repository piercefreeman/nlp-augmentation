from nlp_augmentation.word_substitution.base import WordSubstitutionBase
from nlp_augmentation.base import AugmentationBase
from collections import defaultdict
import numpy as np
from math import log
from nlp_augmentation.data_models import AugmentedDatapoint


class TfIdfWordSubstitution(AugmentationBase, WordSubstitutionBase):
    """TF-IDF Based Word Replacement."""

    def __init__(self, augmentations, token_prob):
        super().__init__(augmentations=augmentations)

        self.token_prob = token_prob
        data_stats = None

    def fit(self, examples):
        if len(examples) < 2:
            raise ValueError("You must fit on at least two examples to calculate corpus statistics.")

        self.data_stats = self.get_data_stats(examples)
        self.idf = self.data_stats["idf"]
        self.tf_idf = self.data_stats["tf_idf"]

        tf_idf_items = self.data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])

        self.tf_idf_keys = []
        self.tf_idf_values = []

        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]

        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max() - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (replace_prob / replace_prob.sum() *
                                        self.token_prob * len(all_words))
        return replace_prob

    def __call__(self, examples):
        assert self.data_stats is not None
        return [
            [
                AugmentedDatapoint(
                    identifier=example.identifier,
                    augmented_index=augmented_index,
                    text=self.process_example(example.text)
                )
                for augmented_index in range(self.augmentations)
            ]
            for example in examples
        ]

    def process_example(self, example):
        all_words = example.split()

        try:
            replace_prob = self.get_replace_prob(all_words)
            replaced_words = self.replace_tokens(
                all_words,
                replace_prob[:len(all_words)],
            )

            return " ".join(replaced_words)
        except RuntimeError:
            return " "

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
                cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1

    def get_data_stats(self, examples):
        """
        Compute the IDF score for each word. Then compute the TF-IDF score.

        TODO: Replace with sklearn vectorizers
        """
        word_doc_freq = defaultdict(int)

        # Compute IDF
        for example in examples:
            cur_word_dict = {}
            cur_sent = example.split()

            for word in cur_sent:
                cur_word_dict[word] = 1
            for word in cur_word_dict:
                word_doc_freq[word] += 1

        idf = {}
        for word in word_doc_freq:
            idf[word] = log(len(examples) * 1. / word_doc_freq[word])

        # Compute TF-IDF
        tf_idf = {}
        for i in examples:
            cur_word_dict = {}
            cur_sent = example.split()

            for word in cur_sent:
                if word not in tf_idf:
                    tf_idf[word] = 0
                tf_idf[word] += 1. / len(cur_sent) * idf[word]

        return {
            "idf": idf,
            "tf_idf": tf_idf,
        }

