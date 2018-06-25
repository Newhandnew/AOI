import os
import tensorflow as tf
import math
# from alexnet import alexnet_v2, alexnet_v2_arg_scope, alexnet_my_arg_scope
import mobilenet_v2
from read_tfrecord import get_data_batch, get_record_number
import inception_v2

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('logs_dir', 'test',
                    'Directory to save the checkpoints and training summaries.')
FLAGS = flags.FLAGS


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.logs_dir, '`logs_dir` is missing.'
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    data_dir = 'data'
    tfrecord_train = 'aoi_7_pattern_train.tfrecords'
    load_checkpoint = True
    train_tf_path = os.path.join(data_dir, tfrecord_train)

    crop_size = [224, 224]
    # Learning params
    learning_rate = 0.01
    num_epochs = 500
    batch_size = 128
    num_examples = get_record_number(train_tf_path)
    num_batches = math.ceil(num_examples / float(batch_size))
    total_steps = num_batches * num_epochs
    print('batch number: {}, total steps: {}'.format(num_batches, total_steps))

    pattern_extension = ['sl', '01', '02', '03', '04', '05', '06']
    num_classes = 2
    num_ng_sample = 4120
    num_ok_sample = 15660
    class_ratio = num_ng_sample / (num_ng_sample + num_ok_sample)


    # Launch the graph
    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.summary.scalar('batch_size', batch_size)

        # Load the data
        train_image_batch, train_label_batch = get_data_batch(
            train_tf_path, pattern_extension, crop_size, batch_size, is_training=True, one_hot=False)
        # convert to float batch
        float_image_batch = tf.to_float(train_image_batch)

        # with slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        #     logits, end_points = mobilenet_v2.mobilenet(float_image_batch, num_classes=num_classes)
        with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
            logits, end_points = inception_v2.inception_v2(float_image_batch, num_classes=num_classes, is_training=True)

        # make summaries of every operation in the node
        for layer_name, layer_op in end_points.items():
            tf.summary.histogram(layer_name, layer_op)

        class_weight = tf.constant([[class_ratio, 1 - class_ratio]])
        # weighted_logits = tf.multiply(logits, class_weight)

        # Specify the loss function (outside the model!)
        one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)
        weight_per_label = tf.transpose(tf.matmul(one_hot_label, tf.transpose(class_weight)))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_label, name='softmax')
        weight_loss = tf.multiply(weight_per_label, loss)
        total_loss = tf.reduce_mean(weight_loss)

        # slim.losses.softmax_cross_entropy(logits, one_hot_label)
        # total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Track accuracy and recall
        predictions = tf.argmax(logits, 1)

        # Define the metrics:
        # Recall@5 would make no sense, because we have only 5 classes here
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, train_label_batch),
            'eval/Recall': slim.metrics.streaming_recall(predictions, train_label_batch),
            'eval/Precision': slim.metrics.streaming_precision(predictions, train_label_batch)
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


if __name__ == '__main__':
    tf.app.run()

