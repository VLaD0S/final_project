"""
ToDo: maybe add some config file to which data_loader.py writes the specifications of the dataset
ToDo: Split into more files:
    > loading the information
    > dealing with the training (and graph)
    > managing Tensorflow Serving ?

"""

import os
import tensorflow as tf
import numpy as np


# Loading data -- paths
training = "skin_data_all_training.npz"
validation = "skin_data_all_validation.npz"
test = "skin_data_all_test.npz"

with np.load(training) as data:
    train_features = data["features"]
    train_labels = data["labels"]
    # print(train_features.shape, train_labels.shape)
with np.load(test) as data:
    test_features = data["features"]
    test_labels = data["labels"]
    # print(test_features.shape, test_labels.shape)

with np.load(validation) as data:
    val_features = data["features"]
    val_labels = data["labels"]
    # print(val_features.shape, val_labels.shape)

# determines the labels
"""
The labels order inside the label_list corresponds to one_hot encoded verison,
which is produced by the "diy_one_hot" function.
"""

label_list = []
for label in val_labels:
    if label not in label_list:
        label_list.append(label)


# assume data shape
data_shape = val_features[0].shape

# determine number of labels
num_labels = len(label_list)
label_indices = list(range(num_labels))


# functions
def accuracy(prediction, labels):
    """
    logging?
    :param prediction:
    :param labels:
    :return:
    """
    # print(prediction.shape)
    prediciton_sum = 100.0 * np.sum(np.argmax(prediction, 1))
    label_sum = np.sum(np.argmax(labels, 1)) / prediction.shape[0]
    return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))
            / prediction.shape[0])


def diy_one_hot(labels):
    """
    :param labels: Array of labels, in order.
    :return: a numpy array of size (<Array size>, <number of labels>),
    where <number of labels> is a "One Hot" encoding of the labels.
    The order of the one_hot encodings correspond the the order of the "label_list" variable.
    """
    one_hot_labels = []
    for lb in range(len(labels)):
        one_hot = np.zeros(num_labels)
        for i in range(num_labels):

            if label_list[i] == labels[lb]:
                one_hot[i] = 1
                one_hot_labels.append(one_hot)
    return np.asarray(one_hot_labels)


def reverse_one_hot(labels):
    """
    creates a label
    :param labels:
    :return: returns a numpy array of
    """
    new_label_list = []
    for tensor in labels:
        for encoding in range(len(tensor)):
            if tensor[encoding] == 1:
                new_label_list.append(label_list[encoding])
    return np.asarray(new_label_list)


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



"""
Defining a Graph - move to new file?
"""

# hyper parameters
# the standard deviation range from which the weights are randomly assigned.
stddev_hyparam = 0.5
learn_rate = 0.1
batch_size = 7

placeholder_shape = (batch_size, data_shape[0], data_shape[1], data_shape[2])

train_labels = diy_one_hot(train_labels)
test_labels = diy_one_hot(test_labels)
val_labels = diy_one_hot(val_labels)


graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(np.float32, shape=placeholder_shape)
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_data = tf.constant(test_features)
    tf_valid_data = tf.constant(val_features)

    # weights/biases:
    """
    List of weights and biases.
   
    Within the weight description, the kernel convolution size is also specified.
    i.e [3, 3, 3, 32] means:
        3x3 kernel convolution : [3, 3, _, _]
        input dataset of depth 3 : [_, _, 3, _]
        output dataset of depth 32 : [_, _, _, 32]
        
    Biases are set at 0.  
    MaxPool operations affect tensor depth.
    """

    # layer 1 :: 160 x 160 x 3 :: 80 x 80 x 32 :: S
    l1_depth = 32
    l1_w = tf.Variable(tf.truncated_normal([3, 3, data_shape[2], l1_depth], stddev=stddev_hyparam))
    l1_b = tf.Variable(tf.zeros(l1_depth))

    # layer 2 :: 80 x 80 x 32 :: 80 x 80 x 64 :: S
    l2_depth = 64
    l2_w = tf.Variable(tf.truncated_normal([3, 3, l1_depth, l2_depth], stddev=stddev_hyparam))
    l2_b = tf.Variable(tf.zeros(l2_depth))

    # MaxPool  1 :: 80 x 80 x 64 :: 40 x 40 x 64 (same depth as l2)

    # layer 3 :: 40 x 40 x 64 :: 40 x 40 x 128 :: S
    l3_depth = 128
    l3_w = tf.Variable(tf.truncated_normal([3, 3, l2_depth, l3_depth], stddev=stddev_hyparam))
    l3_b = tf.Variable(tf.zeros(l3_depth))

    # MaxPool 2 :: 40 x 40 x 128 :: 20 x 20 x 128

    # layer 4 :: 20 x 20 x 128 :: 18 x 18 x 256 :: V
    l4_depth = 256
    l4_w = tf.Variable(tf.truncated_normal([3, 3, l3_depth, l4_depth], stddev=stddev_hyparam))
    l4_b = tf.Variable(tf.zeros(l4_depth))

    # MaxPool 3 :: 18 x 18 x 256 :: 9 x 9 x 256

    # layer 5 :: 9 x 9 x 256 :: 7 x 7 x 512 :: V
    l5_depth = 512
    l5_w = tf.Variable(tf.truncated_normal([3, 3, l4_depth, l5_depth], stddev=stddev_hyparam))
    l5_b = tf.Variable(tf.zeros(l5_depth))

    # layer 6 :: 7 x 7 x 512 into 1 x 25088 :: 1 x 1 x 1024
    l6_depth = 1024
    l6_w = tf.Variable(tf.truncated_normal([7 * 7 * l5_depth, l6_depth], stddev=stddev_hyparam))
    l6_b = tf.Variable(tf.zeros(l6_depth))

    # Final layer 7 to logits :: 1 x 1 x 1024 :: 10
    l7_depth = num_labels
    l7_w = tf.Variable(tf.truncated_normal([l6_depth, l7_depth], stddev=stddev_hyparam))
    l7_b = tf.Variable(tf.zeros(l7_depth))


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


    def model(input_tensor):
        """
        The Model
        :param input_tensor: Dataset
        :return: a
        """
        input_to_float = tf.cast(input_tensor, tf.float32)
        # layer 1
        conv_1 = conv(input_to_float, l1_w, l1_b, stride=2,)

        # layer 2
        conv_2 = conv(conv_1, l2_w, l2_b,)

        # MaxPool 1
        max_1 = maxpool(conv_2,)

        # layer 3
        conv_3 = conv(max_1, l3_w, l3_b,)

        # MaxPool 2
        max_2 = maxpool(conv_3,)

        # layer 4
        conv_4 = conv(max_2, l4_w, l4_b, pad="VALID")

        # MaxPool 3
        max_3 = maxpool(conv_4, )

        # layer 5
        conv_5 = conv(max_3, l5_w, l5_b, pad="VALID")

        # layer 6
        # reshape data from 7 x 7 x 512 to a tensor of list 1 x 25088.
        # output shape is a 1 x 1024 tensor
        layer_5_shape = conv_5.get_shape().as_list()

        pre_conv6 = tf.reshape(conv_5, [layer_5_shape[0],
                                        layer_5_shape[1] * layer_5_shape[2] * layer_5_shape[3]])

        conv_6 = tf.nn.relu(tf.matmul(pre_conv6, l6_w) + l6_b)

        # dropout 1
        #dropout_1 = tf.nn.dropout(conv_6, keep_prob=0.6, name="Dropout")

        # layer 7 final
        return tf.matmul(conv_6, l7_w) + l7_depth


    """
    Training Computation
    """
    logits = model(tf_train_dataset)
    label_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits)
    loss = tf.reduce_mean(label_logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

    # Predictions
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_data))
    valid_prediction = tf.nn.softmax(model(tf_valid_data))

    # use to save variables so we can pick up later
    saver = tf.train.Saver()

sess = tf.Session()
sess.close()

epochs = 3
num_steps = calculate_steps(train_features.shape[0], batch_size, epochs)

num_steps = 100

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    counter = 0
    print("Initializing:")
    for s in range(num_steps):
        #tf.global_variables_initializer().run()
        offset = (s * batch_size) % train_labels.shape[0]
        batch_data = train_features[offset: (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset: (offset + batch_size), :]

        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels,

        }

        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        print('Minibatch loss at step :', s ," " , l)
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), val_labels))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

sess.close()







