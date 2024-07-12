import tensorflow


def createdataset(root: str):
    rawds = tensorflow.keras.utils.image_dataset_from_directory(
        root,
        image_size=(256, 256),
        seed=6059
    )
    trainds, testvalds = tensorflow.keras.utils.split_dataset(rawds, left_size=0.7, right_size=0.3)
    testds, valds = tensorflow.keras.utils.split_dataset(testvalds, left_size=(2 / 3), right_size=(1 / 3))
    return trainds, testds, valds
