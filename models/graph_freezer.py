import tensorflow as tf

saver = tf.train.import_meta_graph("./test_model.meta", clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()

saver.restore(sess, "./test_model")
output_node = "Final"

output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                input_graph_def,
                                                                output_node.split(','))
output_graph = "./cnn_model.pb"

with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()


