# import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras import regularizers
import tensorflow as tf
import numpy as np
import time


class DefoNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        l = 0.001
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                          input_shape=inputShape, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        # model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation(finalAct))
        # return the constructed network architecture
        return model



class EarlyStoppingAtTimeOut(tf.keras.callbacks.Callback):
    """Stop training when the budget training time is met (the budget runs out).

  """

    def __init__(self, budget_time, start_time, patience=5):
        super(EarlyStoppingAtTimeOut, self).__init__()
        self.patience = patience
        self.time = budget_time
        self.start = start_time
        self.stopped_epoch = 0


    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = 0


    def on_epoch_end(self, epoch, logs=None):
        endtime = time.time() - self.start
        temp = logs.get("val_acc")
        if np.less(self.best, temp):
            self.best = temp
            self.wait = 0
        else:
            self.wait += 1
        if endtime >= self.time or self.wait >= self.patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch
        with open("data/logs.txt", "a+") as f:
            # f.write("The average accuracy for epoch {} is {:7.4f}\n".format(epoch, logs["acc"]))
            f.write("{:d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(epoch, logs["acc"], logs["val_acc"], logs["loss"], logs["val_loss"]))
        f.close()


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))