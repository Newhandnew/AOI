import tensorflow as tf
import os
import time
import numpy as np
import glob
import itertools
import cv2
from crop_image import CropImage
from multi_pattern_process import get_pattern_image_path
from read_xml import get_defect_list_from_xml
# import mobilenet_v1
import inception_v1

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('logs_dir', 'inception_7_pattern',
                    'Directory to load checkpoint')
flags.DEFINE_string('test_dir', '/media/new/A43C2A8E3C2A5C14/Downloads/AOI_dataset/new_folder',
                    'Directory of test data')
flags.DEFINE_string('save_image_dir', 'picture_7_pattern_retrain',
                    'Directory of saved pictures')
flags.DEFINE_string('test_type', 'ng',
                    'test type: ok or ng')
FLAGS = flags.FLAGS


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.logs_dir, '`logs_dir` is missing.'
    assert FLAGS.test_dir, '`test_dir` is missing.'
    assert FLAGS.save_image_dir, '`save_image_dir` is missing.'
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    save_image_dir = FLAGS.save_image_dir
    ok_label = 0
    ng_label = 1
    num_class = 2
    batch_size = 100
    crop_image = CropImage(save_image_dir, num_class)
    ok_image_list_path = os.path.join(save_image_dir, 'ok_image_list')
    ng_image_list_path = os.path.join(save_image_dir, 'ng_image_list')
    ok_image_list = open(ok_image_list_path, 'a+')
    ng_image_list = open(ng_image_list_path, 'a+')
    pattern_extension = ['sl', '01', '02', '03', '04', '05', '06']
    image_extension = 'bmp'
    crop_size = [224, 224]
    series_list = []
    series_extension_name = 'yml'
    target_names = os.path.join(FLAGS.test_dir, '*.' + series_extension_name)
    log_path = glob.glob(target_names)
    for file_path in log_path:
        series_number = os.path.splitext(file_path)[0]
        series_list.append(series_number)
    print(series_list)

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
    # with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    #     logits, _ = mobilenet_v2.mobilenet(tf.to_float(image_tensor), num_classes=num_classes)
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, end_points = inception_v1.inception_v1(float_input_tensor, num_classes=num_class, is_training=False)

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

            incorrect_number = 0
            ok_number = 0
            ng_number = 0

            for series_image_path in series_list:
                pattern_path_list = get_pattern_image_path(series_image_path, pattern_extension, image_extension)
                # image_dir = os.path.join(save_image_dir, str(label))
                image_base_name = os.path.basename(series_image_path)
                # image_list_name = os.path.join(image_dir, image_name)
                start_time = time.time()
                image_array = []
                for pattern_path in pattern_path_list:
                    image = crop_image.crop_ok_image(pattern_path, crop_size)
                    image_array.append(image)

                predict_array = np.empty((0, batch_size), int)
                for i in range(int(len(image) / batch_size)):
                    image_index = i * batch_size
                    tmp_predict = sess.run(predictions, feed_dict={
                        side_light_placeholder: image_array[0][image_index:image_index + batch_size],
                        pattern1_placeholder: image_array[1][image_index:image_index + batch_size],
                        pattern2_placeholder: image_array[2][image_index:image_index + batch_size],
                        pattern3_placeholder: image_array[3][image_index:image_index + batch_size],
                        pattern4_placeholder: image_array[4][image_index:image_index + batch_size],
                        pattern5_placeholder: image_array[5][image_index:image_index + batch_size],
                        pattern6_placeholder: image_array[6][image_index:image_index + batch_size]})
                    predict_array = np.append(predict_array, tmp_predict)
                # print("Prediction: {}, shape: {}".format(predict_array, predict_array.shape))
                incorrect = (predict_array != ok_label)
                wrong_index = np.nonzero(incorrect)[0]
                incorrect_number += len(wrong_index)
                if wrong_index.size == 0:
                    ok_number += 1
                else:
                    ng_number += 1

                if FLAGS.test_type == 'ok':
                    crop_image.save_defect_for_ok_image(wrong_index, pattern_path_list, image_base_name,
                                                        pattern_extension, crop_size)
                    ok_list = wrong_index
                elif FLAGS.test_type == 'ng':
                    defect_list = get_defect_list_from_xml(series_image_path + '_remarked.xml')
                    ok_list, ng_list = crop_image.save_defect_for_ng_image(defect_list, wrong_index,
                                                                           pattern_path_list, image_base_name,
                                                                           pattern_extension, crop_size)
                    # save ng image for training
                    for index in ng_list:
                        wrong_image_list = []
                        for image in image_array:
                            wrong_image_list.append(image[index])

                        image_list = crop_image.save_image_array(wrong_image_list, image_base_name, index,
                                                                 pattern_extension, ng_label)
                        print(image_list)
                        ng_image_list.write('{}\n'.format(image_list))
                # save ok image for training
                for index in ok_list:
                    wrong_image_list = []
                    for image in image_array:
                        wrong_image_list.append(image[index])

                    image_list = crop_image.save_image_array(wrong_image_list, image_base_name, index,
                                                             pattern_extension, ok_label)
                    ok_image_list.write('{}\n'.format(image_list))

                elapsed_time = time.time() - start_time
                print('{} inference elapsed time: {}'.format(image_base_name, elapsed_time))
            print("total test number: {}".format(len(series_list)))
            print("ok number: {}, ng number: {}".format(ok_number, ng_number))
            print("incorrect_number: {}".format(incorrect_number))
            ok_image_list.close()

        else:
            print('No checkpoint found')

            # predict_array = sess.run(predictions)
            # print("Prediction: {}".format(predict_array))


if __name__ == '__main__':
    tf.app.run()
