from dataclasses import dataclass

@dataclass
class Datapoint:
    identifier: int
    text: str


@dataclass
class SentenceDatapoint:
    datapoint_identifier: int
    sentence_index: int
    text: str


@dataclass
class AugmentedDatapoint:
    identifier: int
    augmented_index: int
    text: str


@dataclass
class AugmentedSentenceDatapoint:
    datapoint_identifier: int
    sentence_index: int
    augmented_index: int
    text: str

