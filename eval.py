import tensorflow as tf
import os
import math
from alexnet import alexnet_v2, alexnet_my_arg_scope
from read_tfrecord import get_data_batch, get_record_number

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
    data_dir = 'data'
    tfrecord_test = 'aoi_7_pattern_test.tfrecords'
    test_tf_path = os.path.join(data_dir, tfrecord_test)
    crop_size = [224, 224]
    num_classes = 2
    pattern_extension = ['sl', '01', '02', '03', '04', '05', '06']

    num_examples = get_record_number(test_tf_path)
    batch_size = 64
    num_batches = math.ceil(num_examples / float(batch_size))
    # Load the data
    test_image_batch, test_label_batch = get_data_batch(
        test_tf_path, pattern_extension, crop_size, batch_size, is_training=False, one_hot=False)
    # convert to float batch
    test_image_batch = tf.to_float(test_image_batch)
    # Define the network
    with slim.arg_scope(alexnet_my_arg_scope(is_training=False)):
        logits, _ = alexnet_v2(test_image_batch, num_classes=num_classes, is_training=False)

    predictions = tf.argmax(logits, 1)

    # Choose the metrics to compute:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'accuracy': slim.metrics.accuracy(predictions, test_label_batch),
    # })
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'test/Accuracy': slim.metrics.streaming_accuracy(predictions, test_label_batch),
        'test/mse': slim.metrics.streaming_mean_squared_error(predictions, test_label_batch),
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
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_interval_secs = 10 # How often to run the evaluation.
    slim.evaluation.evaluation_loop(
        '',
        logs_path,
        logs_path,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        summary_op=tf.summary.merge(summary_ops),
        eval_interval_secs=eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()