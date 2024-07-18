import keras


def createdataset(root: str):
    rawds = keras.utils.image_dataset_from_directory(
        root,
        image_size=(128, 192),
        seed=6059
    )
    trainds, testvalds = keras.utils.split_dataset(rawds, left_size=0.7, right_size=0.3)
    testds, valds = keras.utils.split_dataset(testvalds, left_size=(2 / 3), right_size=(1 / 3))
    return trainds, testds, valds
