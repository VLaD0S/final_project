"""
Template model file
IMPORTANT:
If desired that the saved model is to work with the graph_freezer.py and the load_model.py, ensure:
1. The input(feature) placeholder in the graph is to be named "Mul", and have a variable input shape
2. The softmax operation is to be named "final_result".

These are so because the inception model itself has these names for the input and output layers,
therefore it saves extra variable parameters to be managed. Otherwise, if you wish to rename those
layers, ensure you've changed them in graph_freezer.py and load_model.py.

"""

import numpy as np
from image_handling.data_manager import DataManager as Dm
import tensorflow as tf
import os

# name of the model. To change.

model_name = "cnn_model_2"

# <editor-fold desc="Loading the data">
"""
Section of the code responsible for looking for the data in the data folder.
"""
data_root = "data"
training_data_path = ""
validation_data_path = ""
testing_data_path = ""

data_files = os.listdir(os.path.join(os.getcwd(), data_root))
data_files_string = " ".join(str(x) for x in data_files)

print("Found datasets: " + str(data_files))
for data_file in data_files:
    if "train" in data_file:
        training_data_path = os.path.join(data_root, data_file)
    if "test" in data_file:
        testing_data_path = os.path.join(data_root, data_file)
    if "valid" in data_file:
        validation_data_path = os.path.join(data_root, data_file)


_training = Dm(training_data_path)
_validation = Dm(validation_data_path)
_testing = Dm(testing_data_path)
# </editor-fold>


# <editor-fold desc="Helper functions">
# function for managing prediction accuracy
def accuracy(prediction, labels):
    """
    Checks the accuracy of the label predictions against the actual labels
    :param prediction: predictions as one_hot encoded tensor
    :param labels: labels as one_hot encoded tensor
    :return: float, resembling the percentage of the accuracy
    """
    pred = np.argmax(prediction, 1)
    label = np.argmax(labels, 1)
    return (100.0 * np.sum(pred == label)
            / prediction.shape[0])


# function for calculating a reasonable number of steps
def calculate_steps(datasize, batchsize, epochs):
    """
    Determines an appropriate number of steps
    :param datasize: Size of the dataset
    :param batchsize: Batches
    :param epochs: Number of times dataset will be retrained.
    :return:recommended step size
    """
    total_datasize = datasize * epochs

    remainder = datasize % batchsize
    if remainder == 0:
        return int(total_datasize / batchsize)
    else:
        steps = int(total_datasize / batchsize) + 1
        print("Dataset of size: ", datasize)
        print("doesn't divide fully into batches of: ", batchsize)
        print("Remainder: ", remainder, ". Adding an extra step...")
        print("Number of steps: ", steps)
        return steps


# NOT USED
# function for mapping a one_hot encoded label onto a label name.=
def decode_onehot(label):
    # decodes onehot tensor to a label.
    if len(label) == len(_training.get_label_names()):
        for hot in range(len(label)):
            if label[hot] == 1:
                return _training.get_label_names()[hot]

        print("No label returned. Faulty one_hot encoded label")
        return
    else:
        print("Error, size of tensor isn't equal to the number of labels.")
        return
# </editor-fold>


# <editor-fold desc="Hyper Parameters and Data Variables">
# Hyper Parameters
stddev_hyparam = 0.04
learn_rate = 0.0045
batch_size = 76
num_epochs = 2
num_steps = calculate_steps(_training.get_features().shape[0], batch_size, num_epochs)

# Data attributes:
num_labels = len(_training.get_label_names())
feature_shape = _training.get_fshape()
label_shape = _training.get_lshape(True)
data_depth = _training.get_fshape()[2]

# </editor-fold>

# Constructing the neural network model inside the graph.
train_graph = tf.Graph()
with train_graph.as_default():

    # <editor-fold desc="Placeholders and constants">
    """
    Creating placeholders for the graph computations:
    _features will be of the shape: [ batch size, x , y , z ]
        - x  and y represent the width and height of the input tensor.
        - z represents the depth of the input tensor.
    _labels will be of the shape: [ batch size, h]
        - h represents the length of the one_hot list
    """

    _features = tf.placeholder(_training.get_features_dtype(), name="Mul")
    _labels = tf.placeholder(_training.get_labels_dtype(True), name="lab")

    # used for determining model accuracy
    _test_features = tf.constant(_testing.get_features())
    _valid_features = tf.constant(_validation.get_features())
    # </editor-fold>

    # <editor-fold desc="wrappers for the convolution and maxpool functions">
    def conv(tensor, weight, bias, stride=1, pad="SAME"):
        """
        Wrapper function for the convolution operation.

        :param tensor: Input tensor (data)
        :param weight: Weight variable
        :param bias:  Bias variable
        :param stride: Stride for the convolution operation. Default is 1
        :param pad: Type of padding. Default is "SAME". Can be changed to "VALID"
        :return: tensor containing the new operation.
        """
        convolution = tf.nn.conv2d(tensor,
                                   weight,
                                   [1, stride, stride, 1],
                                   padding=pad,
                                   use_cudnn_on_gpu=True,)

        bias_addition = tf.nn.bias_add(convolution, bias)
        return tf.nn.relu(bias_addition)

    def maxpool(tensor, ksz=2, stride=2, pad="SAME"):
        """
        :param tensor: Input Tensor (data)
        :param ksz:  Convolution kernel size: (default to a 2x2 kernel)
        :param stride: Stride of the convolution (defaults to stride 2)
        :param pad: Type of padding. Default is "SAME". Can be changed to "VALID"
        :return: max_pooled tensor
        """
        return tf.nn.max_pool(tensor,
                              ksize=[1, ksz, ksz, 1],
                              strides=[1, stride, stride, 1],
                              padding=pad)
    # </editor-fold>

    # <editor-fold desc="Weights and Biases">
    # weights and biases
    # l1 :: 160 x 160 x 3 :: 160 x 160 x 16 :: S
    l1_depth = 16
    l1_w = tf.Variable(tf.truncated_normal([3, 3, data_depth, l1_depth], stddev=stddev_hyparam))
    l1_b = tf.Variable(tf.zeros(l1_depth))

    # max_pool :: 160 x 160 x 16 :: 80 x 80 x 16

    # l2 :: 80 x 80 x 16 :: 80 x 80 x 32 :: S
    l2_depth = 16
    l2_w = tf.Variable(tf.truncated_normal([3, 3, l1_depth, l2_depth], stddev=stddev_hyparam))
    l2_b = tf.Variable(tf.constant(1.0, shape=[l2_depth]))

    # max_pool :: 80 x 80 x 32 :: 40 x 40 x 32

    # l3 :: 40 x 40 x 32 :: 40 x 40 x 64 :: S
    l3_depth = 32
    l3_w = tf.Variable(tf.truncated_normal([3, 3, l2_depth, l3_depth], stddev=stddev_hyparam))
    l3_b = tf.Variable(tf.constant(2.0, shape=[l3_depth]))

    # max_pool :: 40 x 40 x 64 :: 20 x 20 x 64

    # l4 :: 20 x 20 x 64 :: 16 x 16 x 128 :: V
    l4_depth = 32
    l4_w = tf.Variable(tf.truncated_normal([5, 5, l3_depth, l4_depth], stddev=stddev_hyparam))
    l4_b = tf.Variable(tf.constant(1.0, shape=[l4_depth]))

    # l5 :: 16 x 16 x 128 :: 10 x 10 x 256 :: V
    l5_depth = 32
    l5_w = tf.Variable(tf.truncated_normal([7, 7, l4_depth, l5_depth], stddev=stddev_hyparam))
    l5_b = tf.Variable(tf.constant(2.0, shape=[l5_depth]))

    # max_pool :: 10 x 10 x 256 :: 5 x 5 x 256 :: S

    # l6 :: 5 x 5 x 256 :: 1 x 1 x 512 :: V
    l6_depth = 128
    l6_w = tf.Variable(tf.truncated_normal([5, 5, l5_depth, l6_depth], stddev=stddev_hyparam))
    l6_b = tf.Variable(tf.constant(1.0, shape=[l6_depth]))

    # l7 :: 1 x 1 x 512 :: 1 x 1 x 5 :: S
    l7_depth = num_labels
    l7_w = tf.Variable(tf.truncated_normal([l6_depth, l7_depth], stddev=stddev_hyparam))
    l7_b = tf.Variable(tf.constant(2.0, shape=[l7_depth]))
    # </editor-fold>

    def model(data):
        """
        convolution = tf.nn.conv2d(data,
                                   l1_w,
                                   [1, 1, 1, 1],
                                   padding="SAME",
                                   use_cudnn_on_gpu=True,
                                   name="Mul")

        bias_addition_1 = tf.nn.bias_add(convolution, l1_b)

        hidden = tf.nn.relu(bias_addition_1)
        """

        # l1 160-160 3-16
        convolution = conv(data, l1_w, l1_b)

        # mx 160-80 16-16
        mxpool = maxpool(convolution)

        # l2 80-80 16-32
        convolution = conv(mxpool, l2_w, l2_b)

        # mx 80-40 32-32
        mxpool = maxpool(convolution)

        # l3 40-40 32-64
        convolution = conv(mxpool, l3_w, l3_b)

        # mx 40-20 64-64
        mxpool = maxpool(convolution)

        # l4 20-16 64-128
        convolution = conv(mxpool, l4_w, l4_b, pad="VALID")

        # l5 16-10 128-256
        convolution = conv(convolution, l5_w, l5_b, pad="VALID")

        # mx 10-5 256-256
        mxpool = maxpool(convolution)

        # l6 5-1 256-512
        convolution = conv(mxpool, l6_w, l6_b, pad="VALID")

        dropout = tf.nn.dropout(convolution, keep_prob=0.5)
        convolution_shape = convolution.get_shape()

        pre_final = tf.reshape(dropout,
                               [-1, convolution_shape[3]])

        final = tf.matmul(pre_final, l7_w) + l7_b

        return final

    # <editor-fold desc="last graph computation">
    # getting the loss
    logits = model(_features)
    label_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_labels, logits=logits)
    loss = tf.reduce_mean(label_logits)

    # setting the optimizer to use Stochastic Gradient Descent, and try to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

    # getting the predictions
    train_prediction = tf.nn.softmax(logits, name="final_result")
    test_prediction = tf.nn.softmax(model(_test_features))
    valid_prediction = tf.nn.softmax(model(_valid_features))

    # saving the graph
    saver = tf.train.Saver(reshape=True)
    # </editor-fold>

# the session
with tf.Session(graph=train_graph) as sess:
    tf.global_variables_initializer().run()
    max_validation = 0
    max_test = 0

    print("Total steps:", num_steps)
    for step in range(num_steps):
        offset = (step * batch_size) % (_training.get_features_shape()[0] - batch_size)
        batch_data = _training.get_features()[offset:(offset + batch_size), :, :, :]
        batch_labels = _training.get_onehot_labels()[offset:(offset + batch_size), :]

        feed_dict = {_features: batch_data,
                     _labels: batch_labels}

        print("got this far")
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        batch_accuracy = accuracy(predictions, batch_labels)
        validation_accuracy = accuracy(valid_prediction.eval(), _validation.get_onehot_labels())

        # <editor-fold desc="Printing information whenever there's an increase in the test/validation sets">
        if max_validation < validation_accuracy:
            max_validation = validation_accuracy
            print("Step:", step)
            print("Loss:", float(l))
            print("Batch accuracy:", round(batch_accuracy, 3), "%")
            print("Validation accuracy:", round(validation_accuracy, 3), "%")
            print(" ")

        if (step % 25) == 0:
            print("Step:", step)
            print("Loss:", float(l))
            print("Batch accuracy:", round(batch_accuracy, 3), "%")
            print("Validation accuracy:", round(validation_accuracy, 3), "%")
            print(" ")
        # </editor-fold>

    print("...end of training")
    print("Test data accuracy: ", accuracy(test_prediction.eval(), _testing.get_onehot_labels()), "%")
    print("")

    # Saving the data.
    save_folder = "models"
    save_folder = os.path.join(os.getcwd(), "models")
    save_folder = os.path.join(save_folder, model_name)

    print("saving checkpoint")
    saver.save(sess=sess, save_path=os.path.join(save_folder, model_name))
    labels_save = model_name+".txt"
    labels_save = os.path.join(save_folder, labels_save)

    print("saving labels")

    # write labels to a file
    with open(labels_save, 'w') as the_file:
        for item in _training.get_label_names():
            the_file.write("%s\n" % item)
