import tensorflow as tf
import os
import math
import numpy as np
import cv2
import time
from alexnet import alexnet_v2, alexnet_my_arg_scope
from read_tfrecord import get_data_batch, get_record_number
slim = tf.contrib.slim

data_dir = 'data'
tfrecord_test = 'aoi_test.tfrecords'
test_tf_path = os.path.join(data_dir, tfrecord_test)
logs_path = 'logs/alexnet_new_data' #"logs"
crop_size = [224, 224]
num_classes = 2
output_image_dir = "wrong_images"

test_list = os.path.join(data_dir, 'test_list')

with open(test_list) as f:
    test_list_array = [line.strip() for line in f]

num_examples = get_record_number(test_tf_path)
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
# Load the data
test_image_batch, test_label_batch = get_data_batch(
    test_tf_path, crop_size, batch_size, is_training=False, one_hot=False)
# convert to float batch
test_image_batch = tf.to_float(test_image_batch)
# Define the network
with slim.arg_scope(alexnet_my_arg_scope(is_training=False)):
    logits, _ = alexnet_v2(test_image_batch, num_classes=num_classes, is_training=False)

predict = tf.argmax(logits, 1)
correct_pred = tf.equal(predict, test_label_batch)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session(config=session_config) as sess:
    prev_model = tf.train.get_checkpoint_state(logs_path)
    if prev_model:
        saver.restore(sess, prev_model.model_checkpoint_path)
        print('Checkpoint found, {}'.format(prev_model))
    else:
        print('No checkpoint found')

    for i in range(num_batches):
        start_time = time.time()
        pred, label, images, acc = sess.run([predict, test_label_batch, test_image_batch, accuracy])
        elapsed_time = time.time() - start_time
        # print("Testing Accuracy: {:.4f}, time: {}".format(acc, elapsed_time))

        incorrect = (pred != label)
        incorrect_index = np.nonzero(incorrect)[0]
        for index in incorrect_index:
            # file_name = '{}/{}_{}_{}_{}.jpeg'.format(output_image_dir, i, index, pred[index], label[index])
            file_name = test_list_array[i * batch_size + index].rstrip()
            file_name = os.path.basename(file_name)
            output_path = os.path.join(output_image_dir, file_name)
            pattern_name = output_path + '_pattern.png'
            side_light_name = output_path + '_side.png'
            cv2.imwrite(pattern_name, images[index][:, :, 1])
            cv2.imwrite(side_light_name, images[index][:, :, 0])
            print(file_name, end=' ')


