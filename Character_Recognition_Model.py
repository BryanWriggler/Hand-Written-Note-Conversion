'''
2nd File for Project
Goal for this file is to create a model for character recognition, and train it
with the data got from the previous file

(objective for learning: knowing how to build a model with TF)
'''

#input pipeline
import tensorflow as tf 

def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode image
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, [your_image_height, your_image_width, your_num_channels])
    
    # Decode label
    label = tf.io.decode_raw(example['label'], tf.uint8)
    label = tf.cast(label, tf.string)
    
    return image, label