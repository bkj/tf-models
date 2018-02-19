# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


import tensorflow as tf

import box_list

def gather(boxlist, indices, fields=None, scope=None):
  with tf.name_scope(scope, 'Gather'):
    
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    
    subboxlist = box_list.BoxList(tf.gather(boxlist.get(), indices))
    
    if fields is None:
      fields = boxlist.get_extra_fields()
    
    for field in fields:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      
      subfieldlist = tf.gather(boxlist.get_field(field), indices)
      subboxlist.add_field(field, subfieldlist)
    
    return subboxlist


def non_max_suppression(boxlist, thresh, max_output_size, scope=None):
  with tf.name_scope(scope, 'NonMaxSuppression'):
    
    if not 0 <= thresh <= 1.0:
      raise ValueError('thresh must be between 0 and 1')
    
    if not isinstance(boxlist, box_list.BoxList):
      raise ValueError('boxlist must be a BoxList')
    
    if not boxlist.has_field('scores'):
      raise ValueError('input boxlist must have \'scores\' field')
    
    selected_indices = tf.image.non_max_suppression(boxlist.get(), boxlist.get_field('scores'), max_output_size, iou_threshold=thresh)
    return gather(boxlist, selected_indices)
