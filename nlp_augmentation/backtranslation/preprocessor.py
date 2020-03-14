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

from __future__ import absolute_import, division, print_function

import json
import os
import tempfile

import nltk
import tensorflow as tf


class SplitParagraphs:
    def __init__(self, replicas=1, worker_id=0):
        """
        :param replicas: An argument for parallel preprocessing. For example, when replicas=3, we
            divide the data into three parts, and only process one part according to the worker_id.
        :param worker_id: An argument for parallel preprocessing. See 'replicas' for more details

        """
        self.replicas = replicas
        self.worker_id = worker_id

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

    def divide_data_for_worker(self, contents):
        data_per_worker = len(contents) // self.replicas
        remainder = len(contents) - self.replicas * data_per_worker
        worker_id = self.worker_id
        if worker_id < remainder:
            start = (data_per_worker + 1) * worker_id
            end = (data_per_worker + 1) * (worker_id + 1)
        else:
            start = data_per_worker * worker_id + remainder
            end = data_per_worker * (worker_id + 1) + remainder
        if worker_id == self.replicas - 1:
            assert end == len(contents)
        tf.logging.info("processing data from {:d} to {:d}".format(start, end))
        contents = contents[start: end]
        return contents

    def __call__(self, input_file, doc_len_file, output_file):
        """
        :param input_file: The file to be back translated.
        :param doc_len_file: The directory that stores the information of the splitted paragraph.
        :param output_file: The directory that stores the splitted sentences.

        """
        input_file = str(input_file)
        doc_len_file = str(doc_len_file)
        output_file = str(output_file)

        sent_tokenizer = nltk.tokenize.sent_tokenize

        tf.logging.info("loading input data")
        with tf.gfile.Open(input_file) as inf:
            contents = inf.readlines()
        tf.logging.info("finished loading input data")
        print("Contents", contents)
        assert len(contents) >= self.replicas

        contents = self.divide_data_for_worker(contents)

        new_contents = []
        doc_len = []
        # Split paragraphs into sentences since the model is trained on sentence-level
        # translations.
        tf.logging.info("splitting sentence")
        for i in range(len(contents)):
            contents[i] = contents[i].strip()
            if isinstance(contents[i], bytes):
                contents[i] = contents[i].decode("utf-8")
            sent_list = sent_tokenizer(contents[i])
            has_long = False
            if i % 100 == 0:
                tf.logging.info("splitting sentence {:d}".format(i))
            for split_punc in [".", ";", ",", " ", ""]:
                if split_punc == " " or not split_punc:
                    offset = 100
                else:
                    offset = 5
                has_long = False
                new_sent_list = []
                for sent in sent_list:
                    if len(sent) < 300:
                        new_sent_list += [sent]
                    else:
                        has_long = True
                        sent_split = split_sent_by_punc(sent, split_punc, offset)
                        new_sent_list += sent_split
                sent_list = new_sent_list
                if not has_long:
                    break

            # free up memory
            contents[i] = None
            doc_len += [len(sent_list)]
            #  nltk.sent_tokenize in python2 will omit some unicode characters
            for st in sent_list:
                new_contents += [st]

        tf.logging.info("finished spliting paragraphs")

        with tf.gfile.Open(output_file, "w") as ouf:
            for st in new_contents:
                ouf.write(st + "\n")
        with tf.gfile.Open(doc_len_file, "w") as ouf:
            json.dump(doc_len, ouf)
