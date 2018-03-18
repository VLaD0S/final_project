from data_unpacker import DataManager as Dm
import tensorflow as tf
import numpy as np

# loading the data
_training = Dm("skin_data_all_training.npz")
_validation = Dm("skin_data_all_validation.npz")
_testing = Dm("skin_data_all_test.npz")

# training data placeholders
tf_training_features_placeholder = tf.placeholder(_training.get_features_dtype(), _training.get_features_shape())
tf_training_labels_placeholder = tf.placeholder(_training.get_labels_dtype(), _training.get_labels_shape())
tf_training_dataset_placeholder = tf.data.Dataset.from_tensor_slices((tf_training_features_placeholder,
                                                                      tf_training_labels_placeholder))

# initializing the testing and validation functions.
test_features_placeholder = tf.constant(_testing.get_features())
validation_features_placeholder = tf.constant(_validation.get_features())

# initializing the iterator and and the feed dictionary.
iterator = tf_training_dataset_placeholder.make_initializable_iterator()
feed_dictionary = {tf_training_features_placeholder: _training.get_features(),
                   tf_training_labels_placeholder: _training.get_labels()}


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict=feed_dictionary)

