import tensorflow as tf
import os
import numpy as np
import time
from alexnet import alexnet_v2, alexnet_my_arg_scope
from crop_image import CropImage

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
    img_path = '6P8B10G0CLZZ/6P8B10G0CLZZ5.tif'
    crop_size = [224, 224]
    num_classes = 2
    crop_image = CropImage('ng', num_classes)
    test_images = crop_image.crop_ok_image(img_path, crop_size)
    test_images = np.array(test_images)
    test_images_expanded = np.expand_dims(test_images, -1)
    print(test_images_expanded.shape)
    # convert to float batch
    # test_image_batch = tf.to_float(test_images_expanded)

    image_tensor = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1], 1], name='InputImage')
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
            elapsed_time = time.time() - start_time
            print('Checkpoint found, {}'.format(prev_model))
            print('restore elapsed time: {}'.format(elapsed_time))
            start_time = time.time()
            predict_array = sess.run(predictions, feed_dict={image_tensor: test_images_expanded})
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