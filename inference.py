import tensorflow as tf
import os
import numpy as np
import time
from alexnet import alexnet_v2, alexnet_my_arg_scope
from crop_image import CropImage

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
    img_path = '/home/new/Downloads/dataset/Remark_OK/4A833K59QNZZ' #'/media/new/A43C2A8E3C2A5C14/Downloads/AOI_dataset/Remark_OK/4A833K76FTZZ'
    pattern_name = img_path + "_01.bmp"
    side_light_name = img_path + "_sl.bmp"
    crop_size = [224, 224]
    num_classes = 2
    crop_image = CropImage('ng', num_classes)
    pattern_images = crop_image.crop_ok_image(pattern_name, crop_size)
    side_light_images = crop_image.crop_ok_image(side_light_name, crop_size)

    pattern_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern_input')
    side_light_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='side_light_input')
    merged_image = tf.stack((side_light_placeholder, pattern_placeholder), -1)
    float_input_tensor = tf.to_float(merged_image)
    # Define the network
    with slim.arg_scope(alexnet_my_arg_scope(is_training=False)):
        logits, _ = alexnet_v2(float_input_tensor, num_classes=num_classes, is_training=False)

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
            elapsed_time = time.time() - start_time
            print('Checkpoint found, {}'.format(prev_model))
            print('restore elapsed time: {}'.format(elapsed_time))
            start_time = time.time()
            predict_array = sess.run(predictions, feed_dict={side_light_placeholder: side_light_images,
                                                             pattern_placeholder: pattern_images})
            elapsed_time = time.time() - start_time
            crop_image.save_defect_image(predict_array, 'ng.jpg', crop_size)
            print("Prediction: {}, shape: {}".format(predict_array, predict_array.shape))
            print('inference elapsed time: {}'.format(elapsed_time))

        else:
            print('No checkpoint found')

            # predict_array = sess.run(predictions)
            # print("Prediction: {}".format(predict_array))


if __name__ == '__main__':
    tf.app.run()