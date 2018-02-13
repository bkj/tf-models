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
import tensorflow as tf

from delf import datum_io, delf_v1, delf_config_pb2
from ..external import box_list, box_list_ops

def NormalizePixelValues(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
  image = tf.to_float(image)
  image = tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)
  return image


def CalculateReceptiveBoxes(height, width, rf, stride, padding):
  x, y = tf.meshgrid(tf.range(width), tf.range(height))
  coordinates = tf.reshape(tf.stack([y, x], axis=2), [-1, 2])
  
  # [y,x,y,x]
  point_boxes = tf.to_float(tf.concat([coordinates, coordinates], 1))
  bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
  rf_boxes = stride * point_boxes + bias
  return rf_boxes


def CalculateKeypointCenters(boxes):
  return tf.divide(tf.add(tf.gather(boxes, [0, 1], axis=1), tf.gather(boxes, [2, 3], axis=1)), 2.0)


def ExtractKeypointDescriptor(image, layer_name, image_scales, iou, max_feature_num, abs_thres, model_fn):
  """Extract keypoint descriptor for input image.

  Args:
    image: A image tensor with shape [h, w, channels].
    layer_name: The endpoint of feature extraction layer.
    image_scales: A 1D float tensor which contains the scales.
    iou: A float scalar denoting the IOU threshold for NMS.
    max_feature_num: An int tensor denoting the maximum selected feature points.
    abs_thres: A float tensor denoting the score threshold for feature
      selection.
    model_fn: Model function. Follows the signature:

      * Args:
        * `images`: Image tensor which is re-scaled.
        * `normalized_image`: Whether or not the images are normalized.
        * `reuse`: Whether or not the layer and its variables should be reused.

      * Returns:
        * `attention`: Attention score after the non-linearity.
        * `feature_map`: Feature map obtained from the ResNet model.

  Returns:
    boxes: [N, 4] float tensor which denotes the selected receptive box. N is
      the number of final feature points which pass through keypoint selection
      and NMS steps.
    feature_scales: [N] float tensor. It is the inverse of the input image
      scales such that larger image scales correspond to larger image regions,
      which is compatible with scale-space keypoint detection convention.
    features: [N, depth] float tensor with feature descriptors.
    scores: [N, 1] float tensor denoting the attention score.

  Raises:
    ValueError: If the layer_name is unsupported.
  """
  
  original_image_shape_float = tf.gather(tf.to_float(tf.shape(image)), [0, 1])
  image_tensor = NormalizePixelValues(image)
  image_tensor = tf.expand_dims(image_tensor, 0, name='image/expand_dims')

  # Feature depth and receptive field parameters for each network version.
  if layer_name == 'resnet_v1_50/block3':
    feature_depth = 1024
    rf, stride, padding = [291.0, 32.0, 145.0]
  elif layer_name == 'resnet_v1_50/block4':
    feature_depth = 2048
    rf, stride, padding = [483.0, 32.0, 241.0]
  else:
    raise ValueError('Unsupported layer_name.')

  def _ProcessSingleScale(scale_index, boxes, features, scales, scores, reuse=True):
    """Resize the image and run feature extraction and keypoint selection.

       This function will be passed into tf.while_loop() and be called
       repeatedly. The input boxes are collected from the previous iteration
       [0: scale_index -1]. We get the current scale by
       image_scales[scale_index], and run image resizing, feature extraction and
       keypoint selection. Then we will get a new set of selected_boxes for
       current scale. In the end, we concat the previous boxes with current
       selected_boxes as the output.

    Args:
      scale_index: A valid index in the image_scales.
      boxes: Box tensor with the shape of [N, 4].
      features: Feature tensor with the shape of [N, depth].
      scales: Scale tensor with the shape of [N].
      scores: Attention score tensor with the shape of [N].
      reuse: Whether or not the layer and its variables should be reused.

    Returns:
      scale_index: The next scale index for processing.
      boxes: Concatenated box tensor with the shape of [K, 4]. K >= N.
      features: Concatenated feature tensor with the shape of [K, depth].
      scales: Concatenated scale tensor with the shape of [K].
      scores: Concatenated attention score tensor with the shape of [K].
    """
    
    scale = tf.gather(image_scales, scale_index)
    new_image_size = tf.to_int32(tf.round(original_image_shape_float * scale))
    resized_image = tf.image.resize_bilinear(image_tensor, new_image_size)
    
    attention, feature_map = model_fn(resized_image, normalized_image=True, reuse=reuse)
    
    rf_boxes = CalculateReceptiveBoxes(tf.shape(feature_map)[1], tf.shape(feature_map)[2], rf, stride, padding)
    rf_boxes = tf.divide(rf_boxes, scale)
    
    # Re-project back to the original image space.
    attention   = tf.reshape(attention, [-1])
    feature_map = tf.reshape(feature_map, [-1, feature_depth])
    
    # Use attention score to select feature vectors.
    indices           = tf.reshape(tf.where(attention >= abs_thres), [-1])
    selected_boxes    = tf.gather(rf_boxes, indices)
    selected_features = tf.gather(feature_map, indices)
    selected_scores   = tf.gather(attention, indices)
    selected_scales   = tf.ones_like(selected_scores, tf.float32) / scale
    
    # Concat with the previous result from different scales.
    boxes    = tf.concat([boxes, selected_boxes], 0)
    features = tf.concat([features, selected_features], 0)
    scales   = tf.concat([scales, selected_scales], 0)
    scores   = tf.concat([scores, selected_scores], 0)
    
    return scale_index + 1, boxes, features, scales, scores

  output_boxes    = tf.zeros([0, 4], dtype=tf.float32)
  output_features = tf.zeros([0, feature_depth], dtype=tf.float32)
  output_scales   = tf.zeros([0], dtype=tf.float32)
  output_scores   = tf.zeros([0], dtype=tf.float32)

  # Process the first scale separately, the following scales will reuse the
  # graph variables.
  (_, output_boxes, output_features, output_scales, output_scores) = _ProcessSingleScale(
     0,
     output_boxes,
     output_features,
     output_scales,
     output_scores,
     reuse=False
  )
  
  i          = tf.constant(1, dtype=tf.int32)
  num_scales = tf.shape(image_scales)[0]
  keep_going = lambda j, boxes, features, scales, scores: tf.less(j, num_scales)
  
  (_, output_boxes, output_features, output_scales, output_scores) = tf.while_loop(
     cond=keep_going,
     body=_ProcessSingleScale,
     loop_vars=[
         i, output_boxes, output_features, output_scales, output_scores
     ],
     shape_invariants=[
         i.get_shape(),
         tf.TensorShape([None, 4]),
         tf.TensorShape([None, feature_depth]),
         tf.TensorShape([None]),
         tf.TensorShape([None])
     ],
     back_prop=False
  )
  
  feature_boxes = box_list.BoxList(output_boxes)
  feature_boxes.add_field('features', output_features)
  feature_boxes.add_field('scales', output_scales)
  feature_boxes.add_field('scores', output_scores)
  
  nms_max_boxes = tf.minimum(max_feature_num, feature_boxes.num_boxes())
  final_boxes = box_list_ops.non_max_suppression(feature_boxes, iou, nms_max_boxes)
  
  return (
    final_boxes.get(),
    final_boxes.get_field('scales'),
    final_boxes.get_field('features'),
    tf.expand_dims(final_boxes.get_field('scores'), 1)
  )


def BuildModel(layer_name, attention_nonlinear, attention_type, attention_kernel_size):
  def _ModelFn(images, normalized_image, reuse):
    
    image_tensor = images if normalized_image else NormalizePixelValues(images)
    
    model = delf_v1.DelfV1(layer_name)
    _, attention, _, feature_map, _ = model.GetAttentionPrelogit(
        image_tensor,
        attention_nonlinear=attention_nonlinear,
        attention_type=attention_type,
        kernel=[attention_kernel_size, attention_kernel_size],
        training_resnet=False,
        training_attention=False,
        reuse=reuse)
    
    return attention, feature_map
    
  return _ModelFn


def ApplyPcaAndWhitening(data, pca_matrix, pca_mean, output_dim, use_whitening=False, pca_variances=None):
  output = tf.matmul(
      tf.subtract(data, pca_mean),
      tf.slice(pca_matrix, [0, 0], [output_dim, -1]),
      transpose_b=True,
      name='pca_matmul')
  
  if use_whitening:
    output = tf.divide(output, tf.sqrt(tf.slice(pca_variances, [0], [output_dim])), name='whitening')
    
  return output


def DelfFeaturePostProcessing(boxes, descriptors, config):
  """ compute centers of receptive fields; apply PCA + whitening to features """
  
  locations = CalculateKeypointCenters(boxes)
  
  with tf.variable_scope('postprocess'):
    final_descriptors = tf.nn.l2_normalize(descriptors, axis=1, name='l2_normalization')
    
    if config.delf_local_config.use_pca:
      pca_dim = config.delf_local_config.pca_parameters.pca_dim
      
      mean_path = os.path.expanduser(config.delf_local_config.pca_parameters.mean_path)
      pca_mean = tf.constant(datum_io.ReadFromFile(mean_path), dtype=tf.float32)
      
      projection_matrix_path = os.path.expanduser(config.delf_local_config.pca_parameters.projection_matrix_path)
      pca_matrix = tf.constant(datum_io.ReadFromFile(projection_matrix_path), dtype=tf.float32)
      
      if config.delf_local_config.pca_parameters.use_whitening:
        pca_variances_path = os.path.expanduser(config.delf_local_config.pca_parameters.pca_variances_path)
        pca_variances = tf.constant(datum_io.ReadFromFile(pca_variances_path), dtype=tf.float32)
      else:
        pca_variances = None
      
      final_descriptors = ApplyPcaAndWhitening(final_descriptors, pca_matrix, pca_mean, pca_dim,
          config.delf_local_config.pca_parameters.use_whitening, pca_variances)
      
      final_descriptors = tf.nn.l2_normalize(final_descriptors, axis=1, name='pca_l2_normalization')

  return locations, final_descriptors
