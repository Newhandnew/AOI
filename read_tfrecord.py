import tensorflow as tf
import os
import math


def _parse_function(serialized_example, pattern_extension, image_size, one_hot=True, num_classes=2):
    tfrecord_feature = {'label': tf.FixedLenFeature([], tf.int64)}
    for pattern in pattern_extension:
        tfrecord_feature['img_{}'.format(pattern)] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=tfrecord_feature)

    pattern_array = []
    for pattern in pattern_extension:
        pattern_image = tf.decode_raw(features['img_{}'.format(pattern)], tf.uint8)
        pattern_image = tf.reshape(pattern_image, image_size)
        pattern_array.append(pattern_image)
    image = tf.stack(pattern_array, -1)
    label = features['label']
    if one_hot:
        label = tf.one_hot(indices=label, depth=num_classes)
    return image, label


def get_record_number(tfrecord_path):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_path))


def get_data_batch(tfrecord_path, pattern_extension, image_size, batch_size, is_training=False, one_hot=True, num_classes=2):
    """get data iterator for batch training and testing

    tfrecord_path: path for tfrecord
    batch_size: batch_size
    is_training: dataset repeat and shuffle for training, one iteration for testing
    one_hot: flag for one hot label
    num_class: classification classes
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: _parse_function(x, pattern_extension, image_size, one_hot, num_classes))
    if is_training:
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.shuffle(buffer_size=get_record_number(tfrecord_path))
    dataset = dataset.batch(batch_size)
    # Create a one-shot iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == "__main__":

    data_dir = 'data'
    tfrecord_test = 'aoi_7_pattern_test.tfrecords'
    test_tf_path = os.path.join(data_dir, tfrecord_test)
    logs_path = "logs"
    pattern_extension = ['sl', '01', '02', '03', '04', '05', '06']
    image_size = [224, 224]
    num_classes = 2
    is_training = False
    one_hot = False

    num_examples = get_record_number(test_tf_path)
    print(num_examples)
    batch_size = 32
    num_batches = math.ceil(num_examples / float(batch_size))
    # Load the data
    test_image_batch, test_label_batch = get_data_batch(
        test_tf_path, pattern_extension, image_size, batch_size, is_training, one_hot)

    with tf.Session() as sess:
        for i in range(50):
            img, l = sess.run([test_image_batch, test_label_batch])
            print(img.shape, l)
            # cv2.imwrite('{}_side.png'.format(i), img[23][:, :, 0])
            # cv2.imwrite('{}_pattern.png'.format(i), img[23][:, :, 1])
