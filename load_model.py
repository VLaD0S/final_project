"""
This file is using code from the label_image.py file found in the tensorflow/tensorflow repository.
Link: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/examples/label_image/label_image.py

Thus the apache licence: http://www.apache.org/licenses/LICENSE-2.0 applies.

ToDo: Adapt the file so that it takes in the directory where the graph and labels are found as parameter.
ToDo: Add parameters: model input size:,
ToDo: Clean up.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import argparse
import sys, os

import numpy as np
import tensorflow as tf


# loads the graph - taken from label_image.py
def load_graph(model_file):
    """
    Method taken directly from original file
    :param model_file:
    :return:
    """
    tf_graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        print(f)
        graph_def.ParseFromString(f.read())
    with tf_graph.as_default():
        tf.import_graph_def(graph_def)

    return tf_graph


# loads the labels - taken from label_image.py
def load_labels(label_file):
    """
    Method taken directly from original file
    :param label_file:
    :return:
    """

    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


# converts and re-size the input image to fit the input tensor
def image_to_tensor(file_name, size, normalize=True):
    """
    Expects a path to a jpeg file.
    Converts it to the input tensor.
    :param file_name: file path for the image
    :param size: image vertical/horizontal size
    :param normalize: Decides whether to normalize the image
    :return: a normalized image, ready for inference.
    """

    # hyper parameters for mean and standard deviation
    mean = 128
    std = 128

    # read image
    file_reader = tf.read_file(file_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels=3,)

    # ensure image is of adequate input size.
    dimensions = tf.expand_dims(tf.cast(image_reader, tf.float32), 0)
    image_resize = tf.image.resize_bilinear(dimensions, [size, size])

    # normalizing the image:
    if normalize:
        image_resize = tf.divide(tf.subtract(image_resize, [mean]), [std])

    sess = tf.Session()
    return sess.run(image_resize)


if __name__ == "__main__":

    # resolution used to train the dataset.
    image_size = 160

    # normalize input before passing through graph
    normal = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_name", help="Name of the folder where the graph resides")
    parser.add_argument("--image_path", help="The path of the image to be passed through the graph.")
    args = parser.parse_args()

    # name of the graph to be used for inference
    graph_name = args.graph_name
    image_name = args.image_path

    """
    Since inception models are expected to have input of resolution 299x299,
    if an inception model is used, the input size is reshaped to to 299x299 instead.
    """
    print("wtf...")
    if "inception" in graph_name:
        print("WTF!?")
        image_size = 299
        normal = True

    # getting the image
    image_path = os.path.expanduser(os.path.join(os.getcwd(), args.image_path))
    print(image_path)
    input_image = image_to_tensor(image_path, image_size, normalize=normal)

    # logic to get to the right path.
    models_dir_path = os.path.join(os.getcwd(), "models")

    # determining the path to the graph file and the label file
    graph_path = os.path.join(models_dir_path, graph_name, graph_name+".pb")
    labels_path = os.path.join(models_dir_path, graph_name, graph_name+".txt")
    print(graph_path)
    print(labels_path)
    """ --To Delete
    graph_path = "retrained_inception/output_graph.pb"
    graph_path = "models/cnn_model.pb"
    labels_path = "retrained_inception/output_labels.txt"
    """

    # load graph inside a session
    graph = load_graph(graph_path)

    # since the inception model will always be used in this way :
    input_name = "import/" + "Mul"
    output_name = "import/" + "final_result"
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: input_image})

    # removes extra array
    results = np.squeeze(results)

    labels = load_labels(labels_path)

    for i in range(len(labels)):
        print(str(labels[i])+": "+str(results[i]))

