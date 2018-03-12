import tensorflow as tf
import os
import math
import numpy as np
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

num_examples = get_record_number(test_tf_path)
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))
# Load the data
test_image_batch, test_label_batch = get_data_batch(
    test_tf_path, crop_size, batch_size, is_training=False, one_hot=False)
# convert to float batch
test_image_batch = tf.to_float(test_image_batch)
# Define the network
logits, _ = alexnet_v2(test_image_batch, num_classes=num_classes, is_training=False)

predictions = tf.argmax(logits, 1)

# Choose the metrics to compute:
# names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#     'accuracy': slim.metrics.accuracy(predictions, test_label_batch),
# })
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'test/Accuracy': slim.metrics.streaming_accuracy(predictions, test_label_batch),
    # 'test/Recall@5': slim.metrics.streaming_recall_at_k(logits, test_label, 5),
})
for name, tensor in names_to_updates.items():
    tf.summary.scalar(name, tensor)
# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.items():
  op = tf.summary.scalar(metric_name, metric_value)
  op = tf.Print(op, [metric_value], metric_name)
  summary_ops.append(op)

# Setup the global step.
slim.get_or_create_global_step()
with tf.Session() as sess:
    tf.logging.set_verbosity(tf.logging.INFO)
    output_dir = logs_path # Where the summaries are stored.
    eval_interval_secs = 1 # How often to run the evaluation.
    slim.evaluation.evaluation_loop(
        '',
        logs_path,
        logs_path,
        num_evals=num_batches,
        eval_op=predictions,
        summary_op=tf.summary.merge(summary_ops),
        eval_interval_secs=eval_interval_secs)

