"""
File adapted after:
"http://cv-tricks.com/how-to/freeze-tensorflow-models/"

Loads the graph inside the session
"""

import tensorflow as tf
import os
import argparse


def create_graph(name):
    model_name = name
    path_to_data = os.path.join(os.getcwd(), "models")
    path_to_data = os.path.join(path_to_data, model_name)

    print("Loading graph")

    graph_path = path_to_data + "/" + model_name

    saver = tf.train.import_meta_graph((graph_path + ".meta"), clear_devices=True)
    sess = tf.Session()
    saver.restore(sess, graph_path)

    "Graph loaded"
    output_node = "final_result"
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    input_graph_def,
                                                                    output_node.split(','))

    output_graph = graph_path + ".pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the model/graph to be trained")
    args = parser.parse_args()

    create_graph(args.name)

