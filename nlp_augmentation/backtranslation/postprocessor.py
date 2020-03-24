"""
Compose paraphrased sentences back to paragraphs.

"""
import json

from typing import Iterable, Tuple, List
from nlp_augmentation.data_models import AugmentedSentenceDatapoint, AugmentedDatapoint
from itertools import groupby


class SentToParagraph:
    def __call__(self, sentences: Iterable[AugmentedSentenceDatapoint]) -> Iterable[List[AugmentedDatapoint]]:
        """
        :sentences: Sentences that have been augmented by the backtranslation pipeline

        """
        sentences = sorted(
            sentences,
            key=lambda x: (
                x.datapoint_identifier,
                x.augmented_index,
                x.sentence_index,
            )
        )

        for datapoint_identifier, datapoint_group in groupby(
            sentences,
            lambda x: x.datapoint_identifier
        ):
            yield [
                AugmentedDatapoint(
                    identifier=datapoint_identifier,
                    augmented_index=translation_index,
                    text=" ".join([translation.text.strip() for translation in translation_group])
                )
                for translation_index, translation_group in groupby(
                    datapoint_group,
                    lambda x: x.augmented_index
                )
            ]
