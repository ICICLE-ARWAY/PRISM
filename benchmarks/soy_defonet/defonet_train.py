#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from imutils import paths
import numpy as np
import argparse
import os
from defonet_model import DefoNet, EarlyStoppingAtTimeOut
import time
import random
import tensorflow as tf


tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.list_physical_devices('GPU')
seed = 7
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='data/model', help="path to output model")
ap.add_argument("-d", "--dataset", type=str, default='data/dataset')
ap.add_argument("-e", "--epoches", type=str, default="150")
ap.add_argument("-o", "--output", type=str, default="data/output.txt")
ap.add_argument("-i", "--inputsize", type=str, default="108")
ap.add_argument("-b", "--batchsize", type=str, default="64")
ap.add_argument("-t", "--budgettime", type=str, default="300")
#ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

current_time = time.time()
budget_time = int(args["budgettime"])


def list_images(path):
    img_list = []
    f = os.listdir(path)
    for i in range(len(f)):
        subf_name = path + '/' + f[i]
        subf = os.listdir(subf_name)
        for j in range(len(subf)):
            img_path = subf_name + '/' + subf[j]
            img_list.append(img_path)
    return img_list



# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = int(args["epoches"])
INIT_LR = 1e-4
BS = int(args["batchsize"])
img_size = int(args["inputsize"])
IMAGE_SIZE = (img_size, img_size)
BASE_PATH = args["dataset"]
print(BASE_PATH)
TRAIN_PATH = BASE_PATH + "/training"
#VAL_PATH = BASE_PATH + "/validation"
TEST_PATH = BASE_PATH + "/testing"

trainPaths = list_images(TRAIN_PATH)
totalTrain = len(trainPaths)
#totalVal = len(list_images(VAL_PATH))
totalTest = len(list_images(TEST_PATH))

trainLabels = [int(p.split('/')[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
print(trainLabels)
classTotals = trainLabels.sum(axis=0)
print(classTotals)
classWeight = classTotals.max() / classTotals
class_weight = 1
classWeight[1] = class_weight * classWeight[1]
# classWeight[2] = 64 * classWeight[2]
print(classWeight)
classweight = {0: classWeight[0], 1: classWeight[1]}

# initialize the training training data augmentation object
# trainAug = ImageDataGenerator(rescale=1 / 255.0)
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    validation_split=0.23)

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.23)
testAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=True,
    seed=seed,
    batch_size=BS,
    subset='training')

# initialize the validation generator
#valGen = valAug.flow_from_directory(
#    VAL_PATH,
#    class_mode="categorical",
#    target_size=IMAGE_SIZE,
#    color_mode="rgb",
#    shuffle=True,
#    seed=21,
#    batch_size=BS)
valGen = valAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=True,
    seed=seed,
    batch_size=BS,
    subset='validation'
)

# initialize the testing generator
testGen = testAug.flow_from_directory(
    TEST_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)


start = time.time()
callback = EarlyStopping(monitor='val_acc', patience=8)
model = DefoNet.build(
    width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
    depth=3, classes=2,
    finalAct="sigmoid")


print(model.summary(show_trainable=True))
decay_rate = 1
opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=(INIT_LR * decay_rate) / (NUM_EPOCHS * 1), beta_1=0.9, beta_2=0.999, epsilon=1e-8)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["acc", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# fit the model
#H = model.fit(
#    trainGen,
#    steps_per_epoch=totalTrain // BS,
#    validation_data=valGen,
#    validation_steps=totalVal // BS,
#    workers=10,
#    epochs=NUM_EPOCHS)


H = model.fit(
    trainGen,
    validation_data=valGen,
    class_weight=classweight,
    workers=12,
    epochs=NUM_EPOCHS,
    callbacks=EarlyStoppingAtTimeOut(budget_time, current_time)
    )


with open(args["output"], "a+")as f:
    f.write("==================================================\n")
    f.write("dataset: '{}'\n".format(args["dataset"]))

    # f.write("bad weight: {:.4f}\n".format(classWeight[1]))
    f.write("number of epoches: {:d}\n".format(NUM_EPOCHS))
    f.write("Learning rate: {:f}\n".format(INIT_LR))
    f.write("Decay rate: {:f}\n".format(decay_rate))
    f.write("Defo class weight: {:4f}\n".format(classWeight[1]))
    f.write("Seed: {:d}\n".format(seed))
    f.write("[INFO] evaluating network...\n")
    testGen.reset()

    if totalTest % BS == 0:
        STEPS = totalTest // BS
    else:
        STEPS = totalTest // BS + 1
    predIdxs = model.predict(testGen, steps=STEPS)

    # print(len(predIdxs))
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    f.write(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))

    # save the network to disk
    f.write("[INFO] serializing network to '{}'...\n".format(args["model"]))
    model.save(args["model"])

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testGen.classes, predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    Precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    Recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f.write("confusion matrix: \n")
    f.write("[[ {:d}     {:d}]\n".format(cm[0, 0], cm[0, 1]))
    f.write(" [ {:d}     {:d}]]\n".format(cm[1, 0], cm[1, 1]))
    # f.write(" [ {:d}     {:d}     {:d} ]]\n".format(cm[2, 0], cm[2, 1], cm[2, 2]))
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}\n".format(acc))
    print("Precision: {:.4f}\n".format(Precision))
    print("Recall: {:.4f}\n".format(Recall))
    f.write("acc: {:.4f}\n".format(acc))
    f.write("Precision: {:.4f}\n".format(Precision))
    f.write("Recall: {:.4f}\n".format(Recall))
    end = (time.time() - start) / 60.0
    f.write("training time: {:.4f}\n".format(end))
f.close()

# plot the training loss and accuracy
#N = NUM_EPOCHS
#plt.style.use("ggplot")
#plt.figure(figsize=(16,10))
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
#plt.plot(np.arange(0, N), H.history["precision"], label="train_precision")
#plt.plot(np.arange(0, N), H.history["val_precision"], label="val_precision")
#plt.plot(np.arange(0, N), H.history["recall"], label="train_recall")
#plt.plot(np.arange(0, N), H.history["val_recall"], label="val_recall")
#plt.title("Training Loss and Accuracy on Dataset")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="upper right")
#plt.savefig(args["plot"])




#docker run --gpus all -v $PWD/test:/tmp/test d899b11c0636 python3 defonet_train.py -d=test/dataset
#docker run --gpus all -v $PWD/test:/tmp/data -e d0=data/dataset -e e0=1 d899b11c0636
# docker run --privileged --rm --gpus all -v $PWD/../docker-testbench:/tmp/data -e t0=200 zichenzhang/defonet-train:2.2
# -v absolute path to folder on host: mounted folder in container