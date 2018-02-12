# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Python interface for DelfFeatures proto.

Support read and write of DelfFeatures from/to numpy arrays and file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from delf import feature_pb2
from delf import datum_io
import numpy as np
from six.moves import xrange
import tensorflow as tf


def ArraysToDelfFeatures(locations, scales, descriptors, attention, orientations=None):
  num_features = len(attention)
  assert num_features == locations.shape[0]
  assert num_features == len(scales)
  assert num_features == descriptors.shape[0]

  if orientations is None:
    orientations = np.zeros([num_features], dtype=np.float32)
  else:
    assert num_features == len(orientations)

  delf_features = feature_pb2.DelfFeatures()
  for i in xrange(num_features):
    delf_feature             = delf_features.feature.add()
    delf_feature.y           = locations[i, 0]
    delf_feature.x           = locations[i, 1]
    delf_feature.scale       = scales[i]
    delf_feature.orientation = orientations[i]
    delf_feature.strength    = attention[i]
    delf_feature.descriptor.CopyFrom(datum_io.ArrayToDatum(descriptors[i,]))

  return delf_features


def DelfFeaturesToArrays(delf_features):
  num_features = len(delf_features.feature)
  if num_features == 0:
    return np.array([]), np.array([]), np.array([]), np.array([])

  descriptor_dim = len(datum_io.DatumToArray(delf_features.feature[0].descriptor))
  locations      = np.zeros([num_features, 2])
  scales         = np.zeros([num_features])
  descriptors    = np.zeros([num_features, descriptor_dim])
  attention      = np.zeros([num_features])
  orientations   = np.zeros([num_features])

  for i in xrange(num_features):
    delf_feature    = delf_features.feature[i]
    locations[i, 0] = delf_feature.y
    locations[i, 1] = delf_feature.x
    scales[i]       = delf_feature.scale
    descriptors[i,] = datum_io.DatumToArray(delf_feature.descriptor)
    attention[i]    = delf_feature.strength
    orientations[i] = delf_feature.orientation

  return locations, scales, descriptors, attention, orientations


def SerializeToString(locations, scales, descriptors, attention, orientations=None):
  return ArraysToDelfFeatures(locations, scales, descriptors, attention, orientations).SerializeToString()


def ParseFromString(string):
  delf_features = feature_pb2.DelfFeatures()
  delf_features.ParseFromString(string)
  return DelfFeaturesToArrays(delf_features)


def ReadFromFile(file_path):
  with tf.gfile.FastGFile(file_path, 'rb') as f:
    return ParseFromString(f.read())


def WriteToFile(file_path, locations, scales, descriptors, attention, orientations=None):
  serialized_data = SerializeToString(locations, scales, descriptors, attention, orientations)
  with tf.gfile.FastGFile(file_path, 'w') as f:
    f.write(serialized_data)
