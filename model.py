import tensorflow as tf

slim = tf.contrib.slim


def model(inputs, is_training, dropout_rate, num_classes, scope='Net'):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm):
            net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool1')
            tf.summary.histogram("conv1", net)

            net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool2')
            tf.summary.histogram("conv2", net)

            net = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool3')
            tf.summary.histogram("conv3", net)

            net = slim.conv2d(net, 196, [3, 3], padding='SAME', scope='conv4')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool4')
            tf.summary.histogram("conv4", net)

            net = slim.flatten(net, scope='flatten')
            fc1 = slim.fully_connected(net, 1024, scope='fc1')
            tf.summary.histogram("fc1", fc1)

            net = slim.dropout(fc1, dropout_rate, is_training=is_training, scope='fc1-dropout')
            net = slim.fully_connected(net, num_classes, scope='fc2')

            return net, fc1
