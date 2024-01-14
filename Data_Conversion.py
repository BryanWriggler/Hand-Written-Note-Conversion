import tensorflow as tf

def create_tf_example(image_path, label):
    image = open(image_path, 'rb').read()  # Read the image binary data
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example

# Example usage
tf_record_filename = 'Users/Desktop/Hand-Written-Note-Detection/Hand-Written-Note-Conversion/Data'

#read the csv file to get the image paths and label
import csv

with open('Data/english.csv') as file:
    data = csv.reader(file)


with tf.io.TFRecordWriter(tf_record_filename) as writer:

    for row in data:
        tf_example = create_tf_example('Data/Img/' + row[0], row[1])
        writer.write(tf_example.SerializeToString())



