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
Sentence level augmentations: back translation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import random
from absl import flags

import numpy as np
import tensorflow as tf

from nlp_transformation.augmentation.word import word_level_augment
from utils import raw_data_utils


def replace_with_length_check(
    ori_text, new_text,
    use_min_length,
    use_max_length_diff_ratio):
  """Use new_text if the text length satisfies several constraints."""
  if len(ori_text) < use_min_length or len(new_text) < use_min_length:
    if random.random() < 0.001:
      tf.logging.info(
          "not replacing due to short text: \n\tori: {:s}\n\tnew: {:s}\n".format(
              word_level_augment.filter_unicode(ori_text),
              word_level_augment.filter_unicode(new_text)))
    return ori_text
  length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
  if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
    if random.random() < 0.001:
      tf.logging.info(
          ("not replacing due to too different text length:\n"
           "\tori: {:s}\n\tnew: {:s}\n".format(
               word_level_augment.filter_unicode(ori_text),
               word_level_augment.filter_unicode(new_text))))
    return ori_text
  return new_text


def run_augment(examples, paraphrases):

    """Run back translation."""
    use_min_length = 10
    use_max_length_diff_ratio = 0.5
    tf.logging.info("running bt augmentation")

    aug_examples = []
    aug_cnt = 0
    for i in range(len(examples)):
        ori_example = examples[i]
        text_a = replace_with_length_check(
            ori_example.text_a,
            paraphrases[i * text_per_example],
            use_min_length,
            use_max_length_diff_ratio,
        )
        aug_examples.append(text_a)

    return aug_examples

