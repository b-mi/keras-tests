import os
import tensorflow as tf
import sys

# https://www.tensorflow.org/install/pip#windows-native
# pre win 10 a GTX 1660 Ti je max tensorflow vo verzii 2.10
# Caution: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin
# python -m pip install "tensorflow<2.11"

print('tf: ', tf.__version__)
print('is GPU', len(tf.config.list_physical_devices('GPU'))>0)
print('is_built_with_cuda()', tf.test.is_built_with_cuda())
print('list_physical_devices', tf.config.list_physical_devices('GPU'))
print('nvidia driver: ', '555.85')

# Nvidia Cuda Toolkit version
# nvcc --version 
print('nvidia cuda toolkit ', 'release 11.5, V11.5.119')

print('python', sys.version)
