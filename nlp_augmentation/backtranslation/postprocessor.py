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
Compose paraphrased sentences back to paragraphs.

"""
from __future__ import absolute_import, division, print_function

import json

import tensorflow as tf
from absl import app, flags


class SentToParagraph:
    def __call__(self, input_file, doc_len_file):
        """
        :param input_file: back translated file of sentences.
        :param output_file: paraphrased sentences.
        :param doc_len_file: The file that records the length information.

        """
        input_file = str(input_file)
        doc_len_file = str(doc_len_file)

        with tf.gfile.Open(input_file) as inf:
            sentences = inf.readlines()
        with tf.gfile.Open(doc_len_file) as inf:
            doc_len_list = json.load(inf)

        sentence_index = 0

        for i, sent_num in enumerate(doc_len_list):
            paragraph = ""
            for _ in range(sent_num):
                paragraph += sentences[sentence_index].strip() + " "
                sentence_index += 1
            yield paragraph.strip()
