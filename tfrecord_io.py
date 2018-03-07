import tensorflow as tf
from sklearn.model_selection import train_test_split


def transfer_tfrecord(image, label):
    img_raw = image.tobytes()
    tf_transfer = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    return tf_transfer


def split_dataset_write_tfrecord(writer_train, writer_test, dataset, label, test_ratio):
    train_image, test_image = train_test_split(dataset, test_size=test_ratio)
    for image in train_image:
        tf_transfer = transfer_tfrecord(image, label)
        writer_train.write(tf_transfer.SerializeToString())
    for image in test_image:
        tf_transfer = transfer_tfrecord(image, label)
        writer_test.write(tf_transfer.SerializeToString())
    return len(train_image), len(test_image)


def read_and_decode(filename, image_size):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, image_size)
    img = tf.expand_dims(img, -1)
    label = features['label']
    # label = tf.cast(features['label'], tf.int32)
    return img, label


if __name__ == "__main__":
    # import os
    # import cv2
    # from crop_image import crop_ng_image
    #
    # # write test
    # img_path = '/home/new/Downloads/dataset/AOI/1.25/6P7BCXL2QTZZ/6P7BCXL2QTZZ1.tif'
    # defect_point = (642, 564)
    #
    # batch_height = 224
    # batch_width = 224
    # crop_size = [batch_height, batch_width]
    # crop_number = 10
    #
    # img = cv2.imread(img_path, 0)
    #
    # output_dir = 'output'
    # tfrecord_name = 'AOI_train.tfrecords'
    # output_path = os.path.join(output_dir, tfrecord_name)
    #
    # ng = 1  # replace this
    # writer = tf.python_io.TFRecordWriter(output_path)
    # crop_images = crop_ng_image(img, defect_point, crop_size, crop_number)
    # for image in crop_images:
    #     tf_transfer = transfer_tfrecord(image, ng)
    #     writer.write(tf_transfer.SerializeToString())
    #
    # writer.close()

    # read test
    import os
    output_dir = 'output'
    tfrecord_train = 'AOI_train.tfrecords'
    train_tf_path = os.path.join(output_dir, tfrecord_train)
    crop_size = [224, 224]
    num_classes = 2
    img, label = read_and_decode(train_tf_path, crop_size)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=64, capacity=10000,
                                                    min_after_dequeue=8500)
    one_hot_label = tf.one_hot(indices=label_batch, depth=num_classes)
    print(img, label)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # threads = tf.train.start_queue_runners(sess=sess)
        # for i in range(10):
        #     l_test = sess.run(label)
        #     print(l_test)
        #     img, l = sess.run([img_batch, label_batch])
        #     # l = to_categorical(l, 12)
        #     print(img.shape, l)
        #     # print(val)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(100):
            example, l, one_hot = sess.run([img_batch, label_batch, one_hot_label])  # 在会话中取出image和label
            print('{}, {}, {}'.format(example.shape, l, one_hot.shape))
        coord.request_stop()
        coord.join(threads)