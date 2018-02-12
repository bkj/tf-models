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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from skimage.measure import ransac
from skimage.feature import plot_matches
from skimage.transform import AffineTransform

from delf import feature_io

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Read features.
  locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(args.features_1_path)
  num_features_1 = locations_1.shape[0]
  tf.logging.info("Loaded image 1's %d features" % num_features_1)
  
  locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(args.features_2_path)
  num_features_2 = locations_2.shape[0]
  tf.logging.info("Loaded image 2's %d features" % num_features_2)
  
  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(descriptors_1)
  distances, indices = d1_tree.query(descriptors_2, distance_upper_bound=args.distance_threshold)
  
  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,] for i in range(num_features_2) if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],] for i in range(num_features_2) if indices[i] != num_features_1
  ])
  
  # Perform geometric verification using RANSAC.
  model_robust, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000
  )
  
  tf.logging.info('Found %d inliers' % sum(inliers))
  
  # Visualize correspondences, and save to file.
  fig, ax = plt.subplots()
  img_1 = mpimg.imread(args.image_1_path)
  img_2 = mpimg.imread(args.image_2_path)
  inlier_idxs = np.nonzero(inliers)[0]
  plot_matches(
      ax,
      img_1,
      img_2,
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b'
  )
  ax.axis('off')
  ax.set_title('DELF correspondences')
  plt.savefig(args.output_image)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_1_path', type=str, default='test_images/image_1.jpg')
  parser.add_argument('--image_2_path', type=str, default='test_images/image_2.jpg')
  parser.add_argument('--features_1_path', type=str, default='test_features/image_1.delf')
  parser.add_argument('--features_2_path', type=str, default='test_features/image_2.delf')
  parser.add_argument('--output_image', type=str, default='test_match.png')
  parser.add_argument('--distance_threshold', type=float, default=0.8)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  app.run(main=main)
