import tensorflow as tf
from tensorflow.python.framework import graph_util


flags = tf.app.flags
flags.DEFINE_string("output_node_names", "output_argmax", "use comma to separate different nodes")
flags.DEFINE_string("model_folder", "logs/inception_7_pattern", "model with parameters")
flags.DEFINE_string("output_file", "frozen_model.pb", "output model name, place in --model_folder")

FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.model_folder, "--model_folder necessary"
    assert FLAGS.output_file, "--model_file necessary"
    assert FLAGS.output_node_names, "--output_node_name necessary"
    # 得到 checkpoint 文件夹和相关路径
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + FLAGS.output_file

    # 清除设备信息
    clear_devices = True

    # import saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # 得到 graph def
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # 静态化 graph
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                     FLAGS.output_node_names.split(","))

        # 输出
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("generate {} graph nodes".format(len(output_graph_def.node)))
        # show graph
        print([node.name for node in graph.as_graph_def().node])


if __name__ == '__main__':
    tf.app.run()
