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

"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import argparse
import numpy as np
from hashlib import md5
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app

from delf import delf_config_pb2, feature_extractor, feature_io, feature_pb2

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

def main(*args):
  args = parse_args()
  
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Read list of images.
  image_paths = [path.rstrip() for path in sys.stdin]
  num_images = len(image_paths)
  
  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(args.config_path, 'r') as f:
    text_format.Merge(f.read(), config)
  
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    
    with tf.Session() as sess:
      # Initialize variables.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      
      # Loading model that will be used.
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], config.model_path)
      
      graph = tf.get_default_graph()
      
      input_image              = graph.get_tensor_by_name('input_image:0')
      input_score_threshold    = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales       = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num    = graph.get_tensor_by_name('input_max_feature_num:0')
      boxes                    = graph.get_tensor_by_name('boxes:0')
      raw_descriptors          = graph.get_tensor_by_name('features:0')
      feature_scales           = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      attention = tf.reshape(attention_with_extra_dim, [tf.shape(attention_with_extra_dim)[0]])
      
      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(boxes, raw_descriptors, config)
      
      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      tf.logging.info('Starting to extract DELF features from images...')
      for i in range(num_images):
        
        # Get next image.
        im = sess.run(image_tf)
        
        # If descriptor already exists, skip its computation.
        if args.hash_filenames:
          out_desc_filename = md5(image_paths[i]).hexdigest()
        else:
          out_desc_filename = os.path.splitext(os.path.basename(image_paths[i]))[0]
        
        out_desc_fullpath = os.path.join(args.output_dir, out_desc_filename  + _DELF_EXT)
        tf.logging.info('%s -> %s' % (image_paths[i], out_desc_fullpath))
        
        lookup[image_paths[i]] = out_desc_fullpath
        
        if args.lazy:
          if tf.gfile.Exists(out_desc_fullpath):
            tf.logging.info('Skipping %s', image_paths[i])
            continue
        
        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out, attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image            : im,
                 input_score_threshold  : config.delf_local_config.score_threshold,
                 input_image_scales     : list(config.image_scales),
                 input_max_feature_num  : config.delf_local_config.max_feature_num
             })
        
        serialized_desc = feature_io.WriteToFile(out_desc_fullpath, locations_out, feature_scales_out, descriptors_out, attention_out)
        
      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)
      
      json.dump(lookup, open(args.lookup_path, 'w'))

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config-path', type=str, default='delf_config_example.pbtxt')
  parser.add_argument('--output-dir', type=str, default='test_features')
  parser.add_argument('--lookup-path', type=str, default='lookup.json')
  parser.add_argument('--hash-filenames', action="store_true")
  parser.add_argument('--lazy', action="store_true")
  return parser.parse_args()

if __name__ == '__main__':
  app.run(main=main)
