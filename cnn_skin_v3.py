from data_unpacker import DataManager as Dm
import tensorflow as tf
import numpy as np

#<editor-fold desc="Loading the data">
_training = Dm("skin_data_all_training.npz")
_validation = Dm("skin_data_all_validation.npz")
_testing = Dm("skin_data_all_test.npz")
#</editor-fold>

#<editor-fold desc="Helper functions">
# function for managing prediction accuracy
def accuracy(prediction, labels):
    """
    logging?
    :param prediction:
    :param labels:
    :return:
    """
    pred = np.argmax(prediction, 1)
    label = np.argmax(labels, 1)
    return (100.0 * np.sum(pred == label)
            / prediction.shape[0])


# function for calculating a reasonable number of steps
def calculate_steps(datasize, batchsize, epochs):
    """
    Determines an approrpiate number of steps
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


# function for mapping a one_hot encoded label onto a label name.
def decode_onehot(label):
    # decodes onehot tensor to a label.
    if len(label) == len(_training.get_label_names()):
        returned = False
        for hot in range(len(label)):
            if label[hot] == 1:
                returned = True
                return _training.get_label_names()[hot]
        if not returned:
            print("No label returned. Faulty one_hot encoded label")
    else:
        print("Error, size of tensor isn't equal to the number of labels.")
        return
#</editor-fold>

# <editor-fold desc="Hyper Parameters and Data Variables">
# Hyper Parameters
stddev_hyparam = 0.1
learn_rate = 0.01
batch_size = 25
num_epochs = 1
num_steps = calculate_steps(_training.get_features().shape[0], batch_size, num_epochs)

# Data attributes:
num_labels = len(_training.get_label_names())
feature_shape = _training.get_fshape()
label_shape = _training.get_lshape(True)
data_depth = _training.get_fshape()[2]

#</editor-fold>

# Constructing the neural network model

graph = tf.Graph()
with graph.as_default():

    # <editor-fold desc="Aiding parameter creation">
    """
    Creating placeholders for the graph computations:
    _features will be of the shape: [ batch size, x , y , z ]
        - x  and y represent the width and height of the input tensor.
        - z represents the depth of the input tensor.
    _labels will be of the shape: [ batch size, h]
        - h represents the length of the one_hot list
    """
    _fbatch = [batch_size]
    _lbatch = [batch_size]
    _fbatch.extend(list(_training.get_fshape()))
    _lbatch.extend(list(_training.get_lshape(True)))
    #</editor-fold>

    _features = tf.placeholder(_training.get_features_dtype(), shape=_fbatch)
    _labels = tf.placeholder(_training.get_labels_dtype(True), shape=_lbatch)

    _test_features = tf.constant(_testing.get_features())
    _valid_features = tf.constant(_validation.get_features())



    #<editor-fold desc="wrappers for the convolution and maxpool functions">
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
    #</editor-fold>

    # <editor-fold desc="Weights and Biases">
    # weights and biases
    # l1 :: 40 x 40 x 3 :: 20 x 20 x 16 :: S
    l1_depth = 16
    l1_w = tf.Variable(tf.truncated_normal([5, 5, data_depth, l1_depth], stddev=stddev_hyparam))
    l1_b = tf.Variable(tf.zeros(l1_depth))

    # l2 :: 20 x 20 x 16 :: 18 x 18 x 16 :: V
    l2_depth = 16
    l2_w = tf.Variable(tf.truncated_normal([3, 3, l1_depth, l2_depth], stddev=stddev_hyparam))
    l2_b = tf.Variable(tf.zeros(l2_depth))

    # l3 :: 18 x 18 x 16 :: 16 x 16 x 32 :: V
    l3_depth = 32
    l3_w = tf.Variable(tf.truncated_normal([3, 3, l2_depth, l3_depth], stddev=stddev_hyparam))
    l3_b = tf.Variable(tf.zeros(l3_depth))

    # mx1 :: 16 x 16 x 32 :: 8 x 8 x 32 :: S

    # l4 :: 8 x 8 x 32 :: 6 x 6 x 64 :: V
    l4_depth = 64
    l4_w = tf.Variable(tf.truncated_normal([3, 3, l3_depth, l4_depth], stddev=stddev_hyparam))
    l4_b = tf.Variable(tf.zeros(l4_depth))

    # mx2 :: 6 x 6 x 64 :: 3 x 3 x 64 :: S

    # l5 :: 3 x 3 x 64 :: 1 x 1 x 128 :: V
    l5_depth = 128
    l5_w = tf.Variable(tf.truncated_normal([3, 3, l4_depth, l5_depth], stddev=stddev_hyparam))
    l5_b = tf.Variable(tf.constant(1.0, shape=[l5_depth]))

    # final :: 1 x 1 x 128 :: 1 x 1 x 5 (number of labels)
    l6_depth = num_labels
    l6_w = tf.Variable(tf.truncated_normal([l5_depth, l6_depth], stddev=stddev_hyparam))
    l6_b = tf.Variable(tf.constant(1.0, shape=[l6_depth]))

    # </editor-fold>

    def model(data):

        # l1 40-20
        convolution = conv(data, l1_w, l1_b, stride=2)

        # l2 20-18
        convolution = conv(convolution, l2_w, l2_b, pad="VALID")

        # l3 18-16
        convolution = conv(convolution, l3_w, l3_b, pad="VALID")

        # maxpool 16-8
        mxpool = maxpool(convolution)

        # l4 8-6
        convolution = conv(mxpool, l4_w, l4_b, pad="VALID")

        # maxpool 6-3
        mxpool = maxpool(convolution)

        # l5 3-1 - 128
        convolution = conv(mxpool, l5_w, l5_b, pad="VALID")

        convolution_shape = convolution.get_shape()
        pre_final = tf.reshape(convolution,
                               [convolution_shape[0], convolution_shape[3]])

        return tf.matmul(pre_final, l6_w) + l6_b

    # getting the loss
    logits = model(_features)
    label_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_labels, logits=logits)
    loss = tf.reduce_mean(label_logits)

    # setting the optimizer to use Stochastic Gradient Descent, and try to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(_test_features))
    valid_prediction = tf.nn.softmax(model(_valid_features))


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (_training.get_features_shape()[0] - batch_size)
        batch_data = _training.get_features()[offset:(offset + batch_size), :, :, :]
        batch_labels = _training.get_onehot_labels()[offset:(offset + batch_size), :]
        feed_dict = {_features: batch_data, _labels: batch_labels}
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        print(step, " : ",float(l))