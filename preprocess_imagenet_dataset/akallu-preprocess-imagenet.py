import csv
import os.path
import re
import tarfile

import cv2
import numpy as np
from tqdm import tqdm

"""
The Preprocessor Configuration file must follow the rules described below:
 - The CSV must have no blank spaces except the first row after the first two cells

The Preprocessor Configuration file must follow the format described below:
size: s rows x ? columns

Col 0: Common Name of Classes
Col 1: ID of Classes (must be alphanumeric snake case)
Row 2 -> ?: Subclass ID (must start with n and follow with 8 hindu-arabic numerals)
"""

PREPROCESSOR_CONFIG_CSV_LOCATION = "C:\\Users\\anish\\PycharmProjects\\ICICLE-ML-BENCHMARK\\preprocess_imagenet_dataset\\ImageNet2SDSC20Config.csv"  # Path

# The output folder is where the sub-folders containing the images from the classes will go
OUTPUT_FOLDER_LOCATION = "D:\\ImageNetDB\\processed"  # Path

# The file must end in .tar
ILSVRC2012_LOCATION = "D:\\ImageNetDB\\ILSVRC2012_img_train.tar"  # Path

SYNSET_MAPPING_LOCATION = "D:\\ImageNetDB\\raw\\LOC_synset_mapping.txt"


class ImageClass:
    def __init__(self, fullName: str, nameID: str, subclasses: list[str], numID: int):
        self.subclasses = subclasses
        self.fullName = fullName
        self.nameID = nameID
        self.numID = numID

    def addSubclass(self, subclass: str):
        self.subclasses.append(subclass)


class ImageClassSet:
    def __init__(self):
        self.imageClassDict: dict[str, ImageClass] = {}
        self.subclassToClassMap: dict[str, str] = {}
        self.idToClassMap: dict[int, str] = {}

    def addimageclass(self, imageclass: ImageClass):
        self.imageClassDict[imageclass.nameID] = imageclass
        for val in imageclass.subclasses:
            self.subclassToClassMap[val] = imageclass.nameID
            self.idToClassMap[imageclass.numID] = imageclass.nameID

    def getimageclass(self, imageclassname: str) -> ImageClass:
        return self.imageClassDict[imageclassname]

    def getimageclassfromsubclass(self, subclassid: str) -> ImageClass:
        try:
            return self.getimageclass(self.subclassToClassMap[subclassid])
        except KeyError:
            raise KeyError("The subclass ID does not exist in any of the Image Classes provided!")

    def getimageclassfromnumberid(self, numberid: int):
        try:
            return self.getimageclass(self.idToClassMap[numberid])
        except KeyError:
            raise KeyError("The number ID does not exist for of the Image Classes provided!")


def main():
    # TODO: ADD ARGS TO DEFINE CONSTANTS
    preprocessorconfig = open(PREPROCESSOR_CONFIG_CSV_LOCATION, newline='')
    c = int(preprocessorconfig.readline().split(',')[0])

    imageclasses = ImageClassSet()
    for i in range(0, c):
        configArray = preprocessorconfig.readline().split(',')
        configArray[-1] = configArray[-1].strip()
        fullname = configArray[0]
        nameid = configArray[1]
        subclasses = configArray[2:]
        imageclasses.addimageclass((ImageClass(fullname, nameid, subclasses, i)))

    photoset = tarfile.open(ILSVRC2012_LOCATION)
    processedmembers = []

    with tqdm(total=c) as progressbar:
        for imgClass in imageclasses.imageClassDict.values():
            for subclass in imgClass.subclasses:
                if len(re.sub("n[0-9]{8}", "", subclass)) != 0:
                    raise Exception(f"Value {subclass} for {imgClass.nameID} is invalid!")
                try:
                    member = photoset.getmember(subclass + ".tar")
                except KeyError:
                    raise Exception(subclass + " is not a valid ILSVRC category!")
                photoset.extractall(members=[member], path=OUTPUT_FOLDER_LOCATION + "\\tarfiles")
                processedmembers.append(member)
            progressbar.update()

    with tqdm(total=c) as progressbar:
        for imgClass in imageclasses.imageClassDict.values():
            for subclass in imgClass.subclasses:
                photos = tarfile.open(os.path.join(OUTPUT_FOLDER_LOCATION, "tarfiles", subclass + ".tar"))
                photos.extractall(path=os.path.join(OUTPUT_FOLDER_LOCATION, "images", imgClass.nameID))
            progressbar.update()

    # Renaming all the image files & content-aware crop
    with tqdm(total=c) as progressbar:
        valmap = open(os.path.join(OUTPUT_FOLDER_LOCATION, "val_map.csv"), "w")
        testmap = open(os.path.join(OUTPUT_FOLDER_LOCATION, "test_map.csv"), "w")

        try:
            os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "train"))
            os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "val"))
            os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "test"))
        except FileExistsError:
            pass

        globalcount = 0
        for folder in os.listdir(os.path.join(OUTPUT_FOLDER_LOCATION, "images")):
            folderpath = os.path.join(OUTPUT_FOLDER_LOCATION, "images", folder)
            try:
                os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "train", folder))
                os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "val", folder))
                os.mkdir(os.path.join(OUTPUT_FOLDER_LOCATION, "test", folder))
            except FileExistsError:
                pass

            files = list(os.scandir(folderpath))
            filecount = len(os.listdir(folderpath))
            for i in range(0, filecount):
                # 70% train 10% val 20% test
                file = files[i]

                if i / filecount < 0.7:
                    os.rename(os.path.join(folderpath, file.path.lower()), os.path.join(OUTPUT_FOLDER_LOCATION, "train", folder, f"{globalcount}.jpeg"))
                elif i / filecount < 0.8:
                    os.rename(os.path.join(folderpath, file), os.path.join(OUTPUT_FOLDER_LOCATION, "val", folder, f"{globalcount}.jpeg"))
                    valmap.writelines(f"{globalcount},{folder}\n")
                elif i / filecount < 1:
                    os.rename(os.path.join(folderpath, file), os.path.join(OUTPUT_FOLDER_LOCATION, "test", folder, f"{globalcount}.jpeg"))
                    testmap.writelines(f"{globalcount},{folder}\n")
                globalcount += 1
            progressbar.update()


if __name__ == "__main__":
    main()
