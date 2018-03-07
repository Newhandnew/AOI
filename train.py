"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf
from datetime import datetime
import time

from alexnet import alexnet_v2
from model import model
from tfrecord_io import read_and_decode

slim = tf.contrib.slim

"""
Configuration Part.
"""
data_dir = 'data'
tfrecord_train = 'AOI_train.tfrecords'
load_checkpoint = True
train_tf_path = os.path.join(data_dir, tfrecord_train)

crop_size = [224, 224]
# Learning params
learning_rate = 0.01
num_epochs = 3001
save_step = 500
batch_size = 64

# Network params
dropout_rate = 0.5
num_classes = 2
channel = 1

# How often we want to write the tf.summary data to disk
display_step = 20

checkpoint_dir = "checkpoint"
checkpoint_name = 'model.ckpt'
logs_path = "logs"
# Path for tf.summary.FileWriter and to store model checkpoints
# filewriter_path = "tensorboard"
# checkpoint_path = "checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
# if not os.path.isdir(checkpoint_path):
#     os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, crop_size[0], crop_size[1], channel])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
is_training = tf.placeholder(tf.bool, name='IsTraining')

train_image, train_label = read_and_decode(train_tf_path, crop_size)
train_image_batch, train_label_batch = tf.train.shuffle_batch(
    [train_image, train_label], batch_size=batch_size, capacity=10000, min_after_dequeue=8500)
one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)
# Initialize model
logits, fc1 = model(x, is_training, dropout_rate, num_classes)
# model, end_points = alexnet_v2(x, num_classes=num_classes, is_training=is_training)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar("loss", loss)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Restore model weights from previously saved model
    prev_model = tf.train.get_checkpoint_state(checkpoint_dir)
    if load_checkpoint:
        if prev_model:
            saver.restore(sess, prev_model.model_checkpoint_path)
            print('Checkpoint found, {}'.format(prev_model))
        else:
            print('No checkpoint found')

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    # Keep training until reach max iterations
    for epoch in range(num_epochs):
        batch_x, batch_y = sess.run([train_image_batch, one_hot_label])
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, is_training: True})
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            cost, acc, summary = sess.run([loss, accuracy, merged_summary_op],
                                          feed_dict={x: batch_x,
                                                     y: batch_y,
                                                     is_training: False})
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('epoch {}, training accuracy: {:.4f}, loss: {:.5f}, time: {}'
                  .format(epoch, acc, cost, elapsed_time))
            summary_writer.add_summary(summary, epoch)
        if epoch % save_step == 0:
            # Save model weights to disk
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            save_path = saver.save(sess, checkpoint_path)
            print("Model saved in file: {}".format(save_path))

    # save to log path
    # saver.save(sess, os.path.join(logs_path, "model.ckpt"), 1)
    print("Optimization Finished!")
# score = model.fc8
# # Op for calculating the loss
# with tf.name_scope("cross_entropy"):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
#                                                                   labels=y))
#
# # Train op
# with tf.name_scope("train"):
#     # Create optimizer and apply gradient descent to the trainable variables
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#
# # Add the loss to summary
# tf.summary.scalar('cross_entropy', loss)
#
#
# # Evaluation op: Accuracy of the model
# with tf.name_scope("accuracy"):
#     correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # Add the accuracy to the summary
# tf.summary.scalar('accuracy', accuracy)
#
# # Merge all summaries together
# merged_summary = tf.summary.merge_all()
#
# # Initialize the FileWriter
# writer = tf.summary.FileWriter(filewriter_path)
#
# # Initialize an saver for store model checkpoints
# saver = tf.train.Saver()
#
# # Get the number of training/validation steps per epoch
# train_batches_per_epoch = int(np.floor(train_image.data_size / batch_size))
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
#
# # Start Tensorflow session
# with tf.Session() as sess:
#
#     # Initialize all variables
#     sess.run(tf.global_variables_initializer())
#
#     # Add the model graph to TensorBoard
#     writer.add_graph(sess.graph)
#
#     # Load the pretrained weights into the non-trainable layer
#     model.load_initial_weights(sess)
#
#     print("{} Start training...".format(datetime.now()))
#     print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
#                                                       filewriter_path))
#
#     # Loop over number of epochs
#     for epoch in range(num_epochs):
#
#         print("{} Epoch number: {}".format(datetime.now(), epoch+1))
#
#         # Initialize iterator with the training dataset
#         sess.run(training_init_op)
#
#         for step in range(train_batches_per_epoch):
#
#             # get next batch of data
#             image_batch, label_batch = sess.run(next_batch)
#
#             # And run the training op
#             sess.run(train_op, feed_dict={x: image_batch,
#                                           y: label_batch,
#                                           keep_prob: dropout_rate})
#
#             # Generate summary with the current batch of data and write to file
#             if step % display_step == 0:
#                 s = sess.run(merged_summary, feed_dict={x: image_batch,
#                                                         y: label_batch,
#                                                         keep_prob: 1.})
#
#                 writer.add_summary(s, epoch*train_batches_per_epoch + step)
#
#         # Validate the model on the entire validation set
#         print("{} Start validation".format(datetime.now()))
#         sess.run(validation_init_op)
#         test_acc = 0.
#         test_count = 0
#         for _ in range(val_batches_per_epoch):
#
#             image_batch, label_batch = sess.run(next_batch)
#             acc = sess.run(accuracy, feed_dict={x: image_batch,
#                                                 y: label_batch,
#                                                 keep_prob: 1.})
#             test_acc += acc
#             test_count += 1
#         test_acc /= test_count
#         print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
#                                                        test_acc))
#         print("{} Saving checkpoint of model...".format(datetime.now()))
#
#         # save checkpoint of the model
#         checkpoint_name = os.path.join(checkpoint_path,
#                                        'model_epoch'+str(epoch+1)+'.ckpt')
#         save_path = saver.save(sess, checkpoint_name)
#
#         print("{} Model checkpoint saved at {}".format(datetime.now(),
#                                                        checkpoint_name))

# with tf.Graph().as_default():
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.summary.scalar('batch_size', batch_size)
#
#     train_image, train_label = read_and_decode(train_tf_path, crop_size)
#     float_image = tf.to_float(train_image)
#     train_image_batch, train_label_batch = tf.train.shuffle_batch(
#         [float_image, train_label], batch_size=batch_size, capacity=10000, min_after_dequeue=8500)
#
#     # adjust height and width to alexnet dimensions
#     # images: preprocessed images
#     # images_raw: raw images
#     tf.summary.image('image', train_image_batch)
#     # Create the model:
#     # net returns the logit values, end_points returns the nodes / operations
#     # (ordered dict to retain the ordering, very clever!)
#     net, end_points = alexnet_v2(train_image_batch, num_classes=num_classes, is_training=True)
#
#     # make summaries of every operation in the node
#     for layer_name, layer_op in end_points.items():
#         tf.summary.histogram(layer_name, layer_op)
#
#     # Specify the loss function (outside the model!)
#     # one_hot_labels = slim.one_hot_encoding(train_label_batch, num_classes)
#     one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)
#     slim.losses.softmax_cross_entropy(net, one_hot_label)
#     total_loss = slim.losses.get_total_loss()
#
#     # Create some summaries to visualize the training process:
#     tf.summary.scalar('losses/Total Loss', total_loss)
#
#     # Specify the optimizer and create the train op:
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     train_op = slim.learning.create_train_op(total_loss, optimizer)
#
#     # Track accuracy and recall
#     predictions = tf.argmax(net, 1)
#
#     # Define the metrics:
#     # Recall@5 would make no sense, because we have only 5 classes here
#     names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#         'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, train_label_batch),
#         'eval/Recall@2': slim.metrics.streaming_recall_at_k(net, train_label_batch, 2),
#     })
#     for name, tensor in names_to_updates.items():
#         tf.summary.scalar(name, tensor)
#
#     # Run the training:
#     final_loss = slim.learning.train(
#         train_op,
#         logdir=filewriter_path,
#         number_of_steps=num_epochs,
#         session_config=tf.ConfigProto(log_device_placement=True),
#         save_summaries_secs=10
#     )
#
#     print('Finished training. Final batch loss %d' % final_loss)

