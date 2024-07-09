import csv
import os.path
import re
import tarfile
from tqdm import tqdm

"""
The Preprocessor Configuration CSV file must follow the rules described below:
 - The CSV must have no blank spaces except the first row after the first two cells

The Preprocessor Configuration CSV file must follow the format described below:
size: s + 3 rows x c columns

Row 0: [# of subclasses per class - s],[# of classes - c]
Row 1: Common Name of Classes
Row 2: ID of Classes (must be alphanumeric snake case)
Row 3 -> s + 2: Subclass ID (must start with n and follow with 8 hindu-arabic numerals)
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
    preprocessorconfig = csv.reader(open(PREPROCESSOR_CONFIG_CSV_LOCATION, newline=''))
    configarray = list(preprocessorconfig)

    s = int(configarray[0][0])
    c = int(configarray[0][1])

    imageclasses = ImageClassSet()
    for i in range(0, c):
        fullname = configarray[1][i]
        nameid = configarray[2][i]
        subclasses = []
        for j in range(3, s + 3):
            subclasses.append(configarray[j][i])
        imageclasses.addimageclass((ImageClass(fullname, nameid, subclasses, i)))

    photoset = tarfile.open(ILSVRC2012_LOCATION)
    processedmembers = []

    with tqdm(total=(s * c)) as progressbar:
        for row in tqdm(range(3, s + 3)):
            for col in range(0, c):
                subclass = configarray[row][col]
                # Confirm that string is valid
                if len(re.sub("n[0-9]{8}", "", subclass)) != 0:
                    raise Exception(f"Value {subclass} at position " + str(row) + ", " + str(col) + " is invalid!")
                try:
                    member = photoset.getmember(subclass + ".tar")
                except KeyError:
                    raise Exception(subclass + " is not a valid ILSVRC category!")
                photoset.extractall(members=[member], path=OUTPUT_FOLDER_LOCATION + "\\tarfiles")
                processedmembers.append(member)
                progressbar.update()

    # UNPACKING PART TWO
    with tqdm(total=(s * c)) as progressbar:
        for member in processedmembers:
            name = member.name
            photos = tarfile.open(os.path.join(OUTPUT_FOLDER_LOCATION, "tarfiles", name))
            photos.extractall(path=os.path.join(OUTPUT_FOLDER_LOCATION, "images", str(imageclasses.getimageclassfromsubclass(re.sub(".tar", "", name)).numID)))
            progressbar.update()

    # Renaming all the image files
    for folder in os.listdir(os.path.join(OUTPUT_FOLDER_LOCATION, "images")):
        folderpath = os.path.join(OUTPUT_FOLDER_LOCATION, "images", folder)
        counter = 0
        for file in os.scandir(folderpath):
            os.rename(os.path.join(folderpath, file), os.path.join(folderpath, f"{folder}_{counter}.png"))
            counter += 1


if __name__ == "__main__":
    main()
