import tensorflow as tf
import os
import time
from alexnet import alexnet_v2, alexnet_my_arg_scope
from crop_image import CropImage
from multi_pattern_process import get_pattern_image_path

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('logs_dir', 'alexnet_7_pattern_22x22_weight',
                    'Directory to save the checkpoints and training summaries.')
FLAGS = flags.FLAGS


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.logs_dir, '`logs_dir` is missing.'
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    img_path = '/media/new/A43C2A8E3C2A5C14/Downloads/AOI_dataset/Remark_OK/4A833K59DAZZ'#'/home/new/Downloads/dataset/Remark_NG/4A833K74PDZZ' #
    pattern_extension = ['sl', '01', '02', '03', '04', '05', '06']
    image_extension = 'bmp'
    pattern_path_list = get_pattern_image_path(img_path, pattern_extension, image_extension)
    crop_size = [224, 224]
    num_classes = 2
    crop_image = CropImage('ng', num_classes)
    image_array = []
    for pattern_path in pattern_path_list:
        image = crop_image.crop_ok_image(pattern_path, crop_size)
        image_array.append(image)

    side_light_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='side_light_input')
    pattern1_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern1_input')
    pattern2_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern2_input')
    pattern3_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern3_input')
    pattern4_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern4_input')
    pattern5_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern5_input')
    pattern6_placeholder = tf.placeholder(tf.uint8, [None, crop_size[0], crop_size[1]], name='pattern6_input')

    merged_image = tf.stack((side_light_placeholder, pattern1_placeholder, pattern2_placeholder, pattern3_placeholder,
                             pattern4_placeholder, pattern5_placeholder, pattern6_placeholder), -1)
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
            predict_array = sess.run(predictions, feed_dict={side_light_placeholder: image_array[0],
                                                             pattern1_placeholder: image_array[1],
                                                             pattern2_placeholder: image_array[2],
                                                             pattern3_placeholder: image_array[3],
                                                             pattern4_placeholder: image_array[4],
                                                             pattern5_placeholder: image_array[5],
                                                             pattern6_placeholder: image_array[6]})
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