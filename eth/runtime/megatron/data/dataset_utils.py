# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

"""Dataset helpers required by the retained GPT training path."""

import os
import subprocess
import math
import sys


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):
    """Parse weighted dataset prefixes and allocate requested samples."""
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [float(data_prefix[2 * i]) for i in range(num_datasets)]
    prefixes = [data_prefix[2 * i + 1].strip() for i in range(num_datasets)]

    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    datasets_train_valid_test_num_samples = [
        [int(math.ceil(samples * weight * 1.005))
         for samples in train_valid_test_num_samples]
        for weight in weights
    ]
    return prefixes, weights, datasets_train_valid_test_num_samples


def compile_helper():
    """Compile the source-local indexed-dataset helper."""
    path = os.path.abspath(os.path.dirname(__file__))
    result = subprocess.run(["make", "-C", path, f"PYTHON={sys.executable}"])
    if result.returncode != 0:
        raise RuntimeError("making C++ dataset helpers module failed")


def get_train_valid_test_split_(splits_string, size):
    """Convert comma- or slash-separated proportions into split offsets."""
    separator = "," if "," in splits_string else "/"
    splits = [float(value) for value in splits_string.split(separator)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    total = sum(splits)
    assert total > 0.0
    normalized = [split / total for split in splits]

    offsets = [0]
    for index, split in enumerate(normalized):
        offsets.append(offsets[index] + int(round(split * float(size))))
    difference = offsets[-1] - size
    for index in range(1, len(offsets)):
        offsets[index] -= difference
    assert offsets[-1] == size
    return offsets
