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

"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import, division, print_function

import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app

from rsub import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from skimage.measure import ransac
from skimage.feature import plot_matches
from skimage.transform import AffineTransform

from delf import feature_io


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_1_path', type=str, default='queries/surf0.jpg')
  parser.add_argument('--image_2_path', type=str, default='queries/surf2.jpg')
  parser.add_argument('--features_1_path', type=str, default='_results/queries/surf0.delf')
  parser.add_argument('--features_2_path', type=str, default='_results/queries/surf2.delf')
  parser.add_argument('--output_image', type=str, default='test_match.png')
  parser.add_argument('--distance_threshold', type=float, default=0.8)
  return parser.parse_args()


def main(*args):
  
  args = parse_args()
  
  # Read features.
  locations_1, _, descriptors_1, attention_1, _ = feature_io.ReadFromFile(args.features_1_path)
  locations_2, _, descriptors_2, attention_2, _ = feature_io.ReadFromFile(args.features_2_path)
  
  num_features_1 = locations_1.shape[0]
  num_features_2 = locations_2.shape[0]
  
  d1_tree = cKDTree(descriptors_1)
  distances, indices = d1_tree.query(descriptors_2, distance_upper_bound=args.distance_threshold)
  
  has_match          = distances != np.inf
  locations_1_to_use = locations_1[indices[has_match]]
  locations_2_to_use = locations_2[has_match]
  
  _, inliers = ransac(
      (locations_1_to_use, locations_2_to_use), AffineTransform, min_samples=3, residual_threshold=20, max_trials=1000
  )
  
  if inliers is None:
    raise Exception('match_images.py: inliers is None')
  else:
    print('number of inliers -> %d' % len(inliers))
  
  # --
  # Plot
  
  fig, ax = plt.subplots()
  
  inlier_idxs = np.nonzero(inliers)[0]
  plot_matches(
      ax,
      mpimg.imread(args.image_1_path),
      mpimg.imread(args.image_2_path),
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b'
  )
  ax.axis('off')
  ax.set_title('DELF correspondences')
  plt.savefig(args.output_image)


if __name__ == '__main__':
  app.run(main=main)
