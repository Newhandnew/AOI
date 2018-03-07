import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from model import model
from tfrecord_io import read_and_decode

# Parameters
checkpoint_dir = "checkpoint"
checkpoint_name = 'model.ckpt'
output_dir = 'output'
tfrecord_test = 'AOI_test.tfrecords'
test_tf_path = os.path.join(output_dir, tfrecord_test)
crop_size = [224, 224]
test_samples = 2
# Network Parameters
num_classes = 2
channel = 1
test_round = 40

# tf Graph input
x = tf.placeholder(tf.float32, [None, crop_size[0], crop_size[1], channel])
y = tf.placeholder(tf.float32, [None, num_classes])
is_training = tf.placeholder(tf.bool, name='IsTraining')

test_image, test_label = read_and_decode(test_tf_path, crop_size)
train_image_batch, train_label_batch = tf.train.batch(
    [test_image, test_label], batch_size=50, allow_smaller_final_batch=True)
one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)

logits, _ = model(x, is_training, 1, num_classes)
predict = tf.argmax(logits, 1)
label = tf.argmax(y, 1)
correct_pred = tf.equal(predict, label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    prev_model = tf.train.get_checkpoint_state(checkpoint_dir)
    if prev_model:
        saver.restore(sess, prev_model.model_checkpoint_path)
        print('Checkpoint found, {}'.format(prev_model))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        accuracy_array = []
        for i in range(test_round):
            images, labels = sess.run([train_image_batch, one_hot_label])
            predict_array, label_array, acc = sess.run([predict, label, accuracy], feed_dict={
                x: images, y: labels, is_training: False})
            print("Testing Accuracy: {:.4f}".format(acc))
            accuracy_array.append(acc)
            incorrect = (predict_array != label_array)
            incorrect_indices = np.nonzero(incorrect)[0]
            for i, incorrect in enumerate(incorrect_indices[:15]):
                plt.subplot(5, 3, i + 1)
                plt.imshow(images[incorrect].reshape(crop_size), cmap='gray', interpolation='none')
                plt.title("Predicted {}, Class {}".format(predict_array[incorrect], label_array[incorrect]))
                plt.xticks([])
                plt.yticks([])
            plt.savefig('error result {}'.format(i))
        print('accuracy: {}'.format(np.mean(accuracy_array)))
    else:
        print('No checkpoint found')
