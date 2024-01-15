'''
1st File for Project
The goal for this file is to convert the images and labels into something readible
by tensorflow (turning into a data file for training)
'''
import tensorflow as tf

def create_tf_example(image_path, label):
    # Read the image binary data
    image = open(image_path, 'rb').read()

    # Encode the label as bytes
    label_bytes = label.encode('utf-8')

    # Create a feature dictionary
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
    }

    # Create a TensorFlow Example object
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


# Example usage
tf_record_filename = 'DataForTF'

#read the csv file to get the image paths and label
import csv

data = csv.reader(open('english.csv'))


with tf.io.TFRecordWriter(tf_record_filename) as writer:
    for row in data:
        if (row[0] != 'image'):
            tf_example = create_tf_example(row[0], row[1])
            writer.write(tf_example.SerializeToString())



