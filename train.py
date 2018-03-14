import os
import tensorflow as tf
import math
from alexnet import alexnet_v2
from model import model
from read_tfrecord import get_data_batch, get_record_number

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
num_epochs = 500
batch_size = 64
num_examples = get_record_number(train_tf_path)
num_batches = math.ceil(num_examples / float(batch_size))
print('batch number: {}'.format(num_batches))

# Network params
dropout_rate = 0.5
num_classes = 2
channel = 1

checkpoint_dir = "checkpoint_alex"
checkpoint_name = 'model.ckpt'
logs_path = "logs_alex"

# Launch the graph
with tf.Graph().as_default():

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.summary.scalar('batch_size', batch_size)

    # Load the data
    train_image_batch, train_label_batch = get_data_batch(
        train_tf_path, crop_size, batch_size, is_training=True, one_hot=False)
    # convert to float batch
    train_image_batch = tf.to_float(train_image_batch)

    tf.summary.image('image', train_image_batch)

    net, _ = alexnet_v2(train_image_batch, num_classes=num_classes, is_training=True)
    # make summaries of every operation in the node
    # for layer_name, layer_op in end_points.items():
    #     tf.summary.histogram(layer_name, layer_op)

    # Specify the loss function (outside the model!)
    # one_hot_labels = slim.one_hot_encoding(train_label_batch, num_classes)
    one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)
    slim.losses.softmax_cross_entropy(net, one_hot_label)
    total_loss = slim.losses.get_total_loss()

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total Loss', total_loss)

    # Specify the optimizer and create the train op:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Track accuracy and recall
    predictions = tf.argmax(net, 1)

    # Define the metrics:
    # Recall@5 would make no sense, because we have only 5 classes here
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, train_label_batch),
        'eval/Recall@2': slim.metrics.streaming_recall_at_k(net, train_label_batch, 2),
    })
    for name, tensor in names_to_updates.items():
        tf.summary.scalar(name, tensor)
    saver = tf.train.Saver()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.log_device_placement = True
    session = tf.Session(config=session_config)
    prev_model = tf.train.get_checkpoint_state(logs_path)
    if load_checkpoint:
        if prev_model:
            saver.restore(session, prev_model.model_checkpoint_path)
            print('Checkpoint found, {}'.format(prev_model))
        else:
            print('No checkpoint found')
# Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=logs_path,
        number_of_steps=num_epochs * num_batches,
        session_config=session_config,
        save_summaries_secs=20,
        save_interval_secs=300
    )

    print('Finished training. Final batch loss %d' % final_loss)

