import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image
from model import model
from alexnet import alexnet_v2
from read_tfrecord import get_data_batch, get_record_number
slim = tf.contrib.slim

data_dir = 'data'
tfrecord_test = 'AOI_test.tfrecords'
test_tf_path = os.path.join(data_dir, tfrecord_test)
logs_path = "logs_alex"
crop_size = [224, 224]
num_classes = 2
output_image_dir = "wrong_images"

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
        pred, label, images, acc = sess.run([predict, test_label_batch, test_image_batch, accuracy])
        print("Testing Accuracy: {:.4f}".format(acc))

        incorrect = (pred != label)
        incorrect_index = np.nonzero(incorrect)[0]
        for index in incorrect_index:
            file_name = '{}/{}_{}_{}_{}.jpeg'.format(output_image_dir, i, index, pred[index], label[index])
            im = Image.fromarray(images[index].reshape(crop_size))
            im.convert('RGB').save(file_name)

