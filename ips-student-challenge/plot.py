# plot.py - Plot the positions to gain insight into the data
# Author: hanna
# Created: 2022-11-30
#
#
#

import csv
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
z = []

with open("positions.xyz", "r") as file:
  reader = csv.reader(file, delimiter=' ')

  for row in reader:
    x.append(float(row[0]))
    y.append(float(row[1]))
    z.append(float(row[2]))

x = np.array(x)
y = np.array(y)
z = np.array(z)

print("X MIN {} MAX {}".format(np.min(x), np.max(x)))
print("Y MIN {} MAX {}".format(np.min(y), np.max(y)))
print("Z MIN {} MAX {}".format(np.min(z), np.max(z)))

print("Volume is {}".format( (np.max(x) - np.min(x))
                           * (np.max(y) - np.min(y))
                           * (np.max(z) - np.min(z)) ))

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
