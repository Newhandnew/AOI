import os
import tensorflow as tf
import math
# from alexnet import alexnet_v2, alexnet_v2_arg_scope, alexnet_my_arg_scope
import mobilenet_v2
from read_tfrecord import get_data_batch, get_record_number
import inception_v1

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
    tfrecord_train = 'AOI_train.tfrecords'
    load_checkpoint = True
    train_tf_path = os.path.join(data_dir, tfrecord_train)

    crop_size = [224, 224]
    # Learning params
    learning_rate = 0.01
    num_epochs = 500
    batch_size = 256
    num_examples = get_record_number(train_tf_path)
    num_batches = math.ceil(num_examples / float(batch_size))
    print('batch number: {}'.format(num_batches))

    # Network params
    num_classes = 2

    # Launch the graph
    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.summary.scalar('batch_size', batch_size)

        # Load the data
        train_image_batch, train_label_batch = get_data_batch(
            train_tf_path, crop_size, batch_size, is_training=True, one_hot=False)
        # convert to float batch
        float_image_batch = tf.to_float(train_image_batch)

        tf.summary.image('image', float_image_batch)

        # with slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        #     net, end_points = mobilenet_v2.mobilenet(train_image_batch, num_classes=num_classes)
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            logits, end_points = inception_v1.inception_v1(float_image_batch, num_classes=num_classes, is_training=True)

        # make summaries of every operation in the node
        for layer_name, layer_op in end_points.items():
            tf.summary.histogram(layer_name, layer_op)

        # Specify the loss function (outside the model!)
        one_hot_label = tf.one_hot(indices=train_label_batch, depth=num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_label)
        total_loss = slim.losses.get_total_loss()

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
            'eval/Recall@2': slim.metrics.streaming_recall_at_k(logits, train_label_batch, 2),
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

