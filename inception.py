"""ResNet models for Keras.

# Reference paper

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)

"""
"""
-------------------- Modified by JZM --------------------
* Make resnet isolation from keras_application package
* Using keras module directly, rather than supporting get_submodules_from_kwargs
---------------------------------------------------------
"""
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''

from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from inception_common import InceptionV3




def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
