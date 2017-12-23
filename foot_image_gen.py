#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../tf-image-segmentation/')
sys.path.append('slim')

from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input

if __name__ == '__main__':
    number_of_classes = 21
    image_filename = 'nacho_0.jpg'
    image_filename_placeholder = tf.placeholder(tf.string)
    feed_dict = {image_filename_placeholder: image_filename}
    image_tensor = tf.read_file(image_filename_placeholder)
    # tf.shape(image_tensor) = [H, W, CH]
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    # tf.shape(image_batch_tensor) = [1, H, W, CH]
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)
    FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)
    pred, fcn_16s_variable_mapping = FCN_8s(
        image_batch_tensor=image_batch_tensor,
        number_of_classes=number_of_classes,
        is_training=False)
