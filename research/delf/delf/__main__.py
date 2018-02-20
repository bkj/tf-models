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


from __future__ import absolute_import, division, print_function

import os
import sys
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from hashlib import md5
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app

from delf import delf_config_pb2, feature_extractor, feature_io

def main(*args):
  args = parse_args()
  print('delf.main: args=', vars(args), file=sys.stderr)
  
  image_paths = [path.rstrip() for path in sys.stdin]
  
  # Config
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(args.config_path, 'r') as f:
    text_format.Merge(f.read(), config)
  
  with tf.Graph().as_default():
    
    # --
    # IO
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader         = tf.WholeFileReader()
    _, value       = reader.read(filename_queue)
    image_tf       = tf.image.decode_jpeg(value, channels=3)
    
    with tf.Session() as sess:
      
      # --
      # Define graph
      
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], os.path.expanduser(config.model_path))
      
      graph = tf.get_default_graph()
      
      input_image              = graph.get_tensor_by_name('input_image:0')
      input_score_threshold    = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales       = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num    = graph.get_tensor_by_name('input_max_feature_num:0')
      boxes                    = graph.get_tensor_by_name('boxes:0')
      raw_descriptors          = graph.get_tensor_by_name('features:0')
      feature_scales           = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      # resnet = graph.get_tensor_by_name('resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0')
      attention                = tf.reshape(attention_with_extra_dim, [tf.shape(attention_with_extra_dim)[0]])
      
      # for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
      #   print(name, file=sys.stderr)
      
      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(boxes, raw_descriptors, config)
      
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      
      # --
      # Run on images
      
      for image_path in tqdm(image_paths):
        
        if args.hash_filenames:
          outpath = md5(image_path).hexdigest()
        else:
          outpath = os.path.splitext(os.path.basename(image_path))[0]
        
        outpath = os.path.join(args.output_dir, outpath  + '.delf')
        
        img = sess.run(image_tf)
        
        print(json.dumps({"path" : image_path, "key"  : outpath}))
        sys.stdout.flush()
        
        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out, attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image            : img,
                 input_score_threshold  : config.delf_local_config.score_threshold,
                 input_image_scales     : list(config.image_scales),
                 input_max_feature_num  : config.delf_local_config.max_feature_num
             })
        
        # print(resnet_out.shape, file=sys.stderr)
        
        if not args.to_h5:
          _ = feature_io.WriteToFile(outpath, locations_out, feature_scales_out, descriptors_out, attention_out)
        else:
          raise Exception('delf.__main__.py: not implemented yet')
      
      coord.request_stop()
      coord.join(threads)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config-path', type=str, default=os.path.expanduser('~/.delf/delf_config_example.pbtxt'))
  parser.add_argument('--output-dir', type=str, default='_results/delf')
  parser.add_argument('--hash-filenames', action="store_true")
  parser.add_argument('--to-h5', action="store_true")
  return parser.parse_args()


if __name__ == '__main__':
  app.run(main=main)
