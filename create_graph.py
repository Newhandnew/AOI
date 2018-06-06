import tensorflow as tf
import os
import time
from alexnet import alexnet_v2, alexnet_my_arg_scope

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('logs_dir', 'alex_batch_norm_pattern2_batch512',
                    'Directory to save the checkpoints and training summaries.')
FLAGS = flags.FLAGS


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.logs_dir, '`logs_dir` is missing.'
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    crop_size = [224, 224]
    num_classes = 2

    image_tensor = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1], 1], name='input_image')
    # Define the network
    with slim.arg_scope(alexnet_my_arg_scope(is_training=False)):
        logits, _ = alexnet_v2(tf.to_float(image_tensor), num_classes=num_classes, is_training=False)

    predictions = tf.argmax(logits, 1, name='output_argmax')
    # Setup the global step.
    tf.train.get_or_create_global_step()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    tf.logging.set_verbosity(tf.logging.INFO)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=session_config) as sess:
        start_time = time.time()
        prev_model = tf.train.get_checkpoint_state(logs_path)
        if prev_model:
            saver.restore(sess, prev_model.model_checkpoint_path)
            tf.train.write_graph(sess.graph_def, logs_path, "nn_model.pbtxt", as_text=True)
            checkpoint_path = os.path.join(logs_path, "nn+model.ckpt")
            saver.save(sess, checkpoint_path)

            elapsed_time = time.time() - start_time
            print('Checkpoint found, {}'.format(prev_model))
            print('restore elapsed time: {}'.format(elapsed_time))

        else:
            print('No checkpoint found')

            # predict_array = sess.run(predictions)
            # print("Prediction: {}".format(predict_array))




if __name__ == '__main__':
    tf.app.run()