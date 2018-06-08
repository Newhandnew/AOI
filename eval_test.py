import tensorflow as tf
import os
import math
from alexnet import alexnet_v2, alexnet_my_arg_scope
from read_tfrecord import get_data_batch, get_record_number
import time

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('logs_dir', 'alexnet_new_data',
                    'Directory to save the checkpoints and training summaries.')
FLAGS = flags.FLAGS


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.logs_dir, '`logs_dir` is missing.'
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    data_dir = 'data'
    tfrecord_test = 'aoi_train.tfrecords'
    test_tf_path = os.path.join(data_dir, tfrecord_test)
    crop_size = [224, 224]
    num_classes = 2

    num_examples = get_record_number(test_tf_path)
    batch_size = 1000
    num_batches = math.ceil(num_examples / float(batch_size))
    # Load the data
    test_image_batch, test_label_batch = get_data_batch(
        test_tf_path, crop_size, batch_size, is_training=False, one_hot=False)
    # convert to float batch
    test_image_batch = tf.to_float(test_image_batch)
    # Define the network
    with slim.arg_scope(alexnet_my_arg_scope(is_training=False)):
        logits, _ = alexnet_v2(test_image_batch, num_classes=num_classes, is_training=False)

    predictions = tf.argmax(logits, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        start_time = time.time()
        prev_model = tf.train.get_checkpoint_state(logs_path)
        if prev_model:
            saver.restore(sess, prev_model.model_checkpoint_path)
            elapsed_time = time.time() - start_time
            print('Checkpoint found, {}'.format(prev_model))
            print('restore elapsed time: {}'.format(elapsed_time))
            start_time = time.time()
            predict_array = sess.run(predictions)
            elapsed_time = time.time() - start_time
            print("Prediction: {}, shape: {}".format(predict_array, predict_array.shape))
            print('inference elapsed time: {}'.format(elapsed_time))

        else:
            print('No checkpoint found')


if __name__ == '__main__':
    tf.app.run()