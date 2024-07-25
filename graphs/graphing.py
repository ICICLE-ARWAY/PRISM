import csv
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Location of data
DATA_CSV_LOCATION = "./graphs/data.csv"

# Setting desired graph
X_AXIS = "Cores"
Y_AXIS = "Duration (s)"

csvreader = open(DATA_CSV_LOCATION, newline='')
numpoints = int(csvreader.readline().strip())
labeltoid = {"Cores": 0, "Standard Deviation": 1, "Duration (s)": 2, "Average (min)": 3, "Efficiency Index": 4, "Normalized Efficiency": 5, "Speedup": 6}
xpoints = []
ypoints = []


if X_AXIS not in labeltoid or Y_AXIS not in labeltoid:
        print("Invalid labels, please use one of the following:")
        for label in labeltoid: print(label)
        sys.exit()


for i in range(numpoints):
    data = list(map(float, csvreader.readline().strip().split(',')))
    print(len(data))
    xpoints.append(data[labeltoid[X_AXIS]])
    ypoints.append(data[labeltoid[Y_AXIS]])


xpoints = np.array(xpoints)
ypoints = np.array(ypoints)

time.sleep(0)
plt.plot(xpoints, ypoints)
plt.title(f"{X_AXIS} vs. {Y_AXIS}")
plt.ylabel(Y_AXIS)
plt.xlabel(X_AXIS)
plt.show()

    


