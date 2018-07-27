import tensorflow as tf
from tensorflow.python.framework import graph_util
import os


flags = tf.app.flags
flags.DEFINE_string("output_node_names", "output_argmax", "use comma to separate different nodes")
flags.DEFINE_string("logs_dir", "logs/inception_7_pattern", "model with parameters")
flags.DEFINE_string("output_file", "frozen_model.pb", "output model name, place in --model_folder")

FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.logs_dir, "--logs_dir necessary"
    assert FLAGS.output_file, "--model_file necessary"
    assert FLAGS.output_node_names, "--output_node_name necessary"
    log_folder = os.path.join("logs", FLAGS.logs_dir)
    checkpoint = tf.train.get_checkpoint_state(log_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + FLAGS.output_file

    # clear device information
    clear_devices = True

    # import saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                     FLAGS.output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("generate {} graph nodes".format(len(output_graph_def.node)))
        # show graph
        print([node.name for node in graph.as_graph_def().node])


if __name__ == '__main__':
    tf.app.run()
