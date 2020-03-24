# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Split the paragraph into sentences for back translation.

"""
import json
import os
import tempfile

from nltk.tokenize import sent_tokenize
from typing import Iterable
from nlp_augmentation.data_models import Datapoint, SentenceDatapoint


class SplitParagraphs:
    def split_sent_by_punc(self, sent, punc, offset):
        """Further split sentences when nltk's sent_tokenizer fail."""
        sent_list = []
        start = 0

        while start < len(sent):
            if punc:
                pos = sent.find(punc, start + offset)
            else:
                pos = start + offset
            if pos != -1:
                sent_list += [sent[start: pos + 1]]
                start = pos + 1
            else:
                sent_list += [sent[start:]]
                break

        return sent_list

    def __call__(self, paragraphs: Iterable[str]) -> Iterable[SentenceDatapoint]:
        new_contents = []

        # Split paragraphs into sentences since the model is trained on sentence-level
        # translations.
        for paragraph in paragraphs:
            text = paragraph.text.strip()

            if isinstance(text, bytes):
                text = text.decode("utf-8")

            sentence_list = sent_tokenize(text)

            has_long = False

            for split_punc in [".", ";", ",", " ", ""]:
                if split_punc == " " or not split_punc:
                    offset = 100
                else:
                    offset = 5
                has_long = False
                new_sent_list = []
                for sent in sentence_list:
                    if len(sent) < 300:
                        new_sent_list += [sent]
                    else:
                        has_long = True
                        sent_split = self.split_sent_by_punc(sent, split_punc, offset)
                        new_sent_list += sent_split
                sentence_list = new_sent_list
                if not has_long:
                    break

            for i, sentence in enumerate(sentence_list):
                yield SentenceDatapoint(
                    text=sentence,
                    sentence_index=i,
                    datapoint_identifier=paragraph.identifier,
                )
