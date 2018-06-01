import tensorflow as tf
import cv2
import numpy as np
import time


flags = tf.app.flags
flags.DEFINE_string("frozen_model_name", "logs/mobilenet/frozen_model.pb", "frozen name")
FLAGS = flags.FLAGS


def load_graph(frozen_graph_filname):
    with tf.gfile.GFile(frozen_graph_filname, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="inference")
        return graph


def main(_):
    assert FLAGS.frozen_model_name, "--frozen_model_name necessary"
    graph = load_graph(FLAGS.frozen_model_name)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name('inference/input_image:0 ')
    y = graph.get_tensor_by_name('inference/output_argmax:0')
    # print(x, y)
    #
    img_path = '/home/new/PycharmProjects/auto_labeling/check/6P7BCPY1HBZZ_389.png' # 6P7BCNL9GQZZ_393.png # 6P8B1050QKZZ_47.png'
    image = cv2.imread(img_path, 0)
    test_images = np.array(image)
    test_images_expanded = np.expand_dims(test_images, -1)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    # # We launch a Session
    start_time = time.time()
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        y_out = sess.run(y, feed_dict={
            x: [test_images_expanded]  # < 45
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out)  # [[ False ]] Yay, it works!

    elapsed_time = time.time() - start_time
    print('inference elapsed time: {}'.format(elapsed_time))


if __name__ == '__main__':
    tf.app.run()