import keras

def create_datasets_from_images(dtype, data_dir):
    """ Create CIFAR training and test datasets with keras.utils.image_dataset_from_directory """

    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=data_dir+'/train',
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=None,
        image_size=(32, 32),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, dtype), tf.cast(y, tf.uint8)))

    test_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=data_dir+'/test',
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=None,
        image_size=(32, 32),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, dtype), tf.cast(y, tf.uint8)))

    return train_dataset, test_dataset