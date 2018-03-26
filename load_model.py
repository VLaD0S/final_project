"""
This file is using code from the label_image.py file found in the tensorflow/tensorflow repository.
Link: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/examples/label_image/label_image.py

Thus the apache licence: http://www.apache.org/licenses/LICENSE-2.0 applies.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os

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


def print_prediction(graph_nm, image_nm, res, normal=False):
    """
    :param graph_nm: graph name
    :param image_nm: image name
    :param res: resolution of the image
    :param normal: whether the image should be normalized
    :return: label and value of the highest prediction.
    """
    # name of the graph to be used for inference
    graph_name = graph_nm
    image_name = image_nm
    im_res = res
    norm = normal
    """
    Since inception models are expected to have input of resolution 299x299,
    if an inception model is used, the input size is reshaped to to 299x299 instead.
    """

    if "inception" in graph_name:
        im_res = 299
        norm = True

    # getting the image
    image_path = os.path.expanduser(os.path.join(os.getcwd(), image_name))
    input_image = image_to_tensor(image_path, im_res, normalize=norm)

    # logic to get to the right path.
    models_dir_path = os.path.join(os.getcwd(), "models")

    # determining the path to the graph file and the label file
    graph_path = os.path.join(models_dir_path, graph_name, graph_name + ".pb")
    labels_path = os.path.join(models_dir_path, graph_name, graph_name + ".txt")

    # load graph inside a session
    graph = load_graph(graph_path)

    # since the inception model will always be used in this way :
    input_name = "import/" + "Mul"
    output_name = "import/" + "final_result"
    input_op = graph.get_operation_by_name(input_name)
    output_op = graph.get_operation_by_name(output_name)

    # return graph, input_op, output_op, input_image, labels_path

    sess = tf.Session(graph=graph)
    results = sess.run(output_op.outputs[0],
                {input_op.outputs[0]: input_image})

    # removes extra array
    results = np.squeeze(results)

    labels = load_labels(labels_path)

    max_label = 0
    max_name = ""

    for i in range(len(labels)):

        print(str(labels[i]) + ": " + str(results[i]))

        if max_label < results[i]:
            max_label = results[i]
            max_name = labels[i]

    return max_label, max_name


if __name__ == "__main__":

    # resolution used to train the dataset.
    image_size = 160

    # normalize input before passing through graph
    to_normalize = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_name", help="Name of the folder where the graph resides")
    parser.add_argument("--image_path", help="The path of the image to be passed through the graph.")
    args = parser.parse_args()
    var = print_prediction(args.graph_name, args.image_path, image_size, to_normalize)
    # print(var)







