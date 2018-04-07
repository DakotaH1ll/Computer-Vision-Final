
# $$$$$$$\                                $$\
# $$  __$$\                               $$ |
# $$ |  $$ | $$$$$$\   $$$$$$\   $$$$$$\  $$ | $$$$$$\       $$$$$$\  $$\   $$\
# $$$$$$$  |$$  __$$\ $$  __$$\ $$  __$$\ $$ |$$  __$$\     $$  __$$\ $$ |  $$ |
# $$  ____/ $$$$$$$$ |$$ /  $$ |$$ /  $$ |$$ |$$$$$$$$ |    $$ /  $$ |$$ |  $$ |
# $$ |      $$   ____|$$ |  $$ |$$ |  $$ |$$ |$$   ____|    $$ |  $$ |$$ |  $$ |
# $$ |      \$$$$$$$\ \$$$$$$  |$$$$$$$  |$$ |\$$$$$$$\ $$\ $$$$$$$  |\$$$$$$$ |
# \__|       \_______| \______/ $$  ____/ \__| \_______|\__|$$  ____/  \____$$ |
#                               $$ |                        $$ |      $$\   $$ |
#                               $$ |                        $$ |      \$$$$$$  |
#                               \__|                        \__|       \______/


# Dakota Hill
# 100523538
# Computer Vision Final Project

import numpy as np
import cv2
import sys
from random import randint


def non_max_suppression(boxesOld, maxOverlap):

	# No empty arrays
	if len(boxesOld) == 0:
		return []

	# Convert integer values in the array to floats
	if boxesOld.dtype.kind == "i":
		boxesOld = boxesOld.astype("float")

	# Create new array for storing boxes
	boxesNew = []

	# Get the coordinates of the old boxes
	x1 = boxesOld[:,0]
	y1 = boxesOld[:,1]
	x2 = boxesOld[:,2]
	y2 = boxesOld[:,3]

	# Calculate area and sort the boxes by bottom right corner Y value
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# Loop until no more boxes
	while len(idxs) > 0:
		# Add the indexes in descending order to the new array
		last = len(idxs) - 1
		i = idxs[last]
		boxesNew.append(i)

		# Find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# Calculate new width and heigt of the new bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# Compute the overlap between boxes
		overlap = (w * h) / area[idxs[:last]]

		# If there's more overlap than the allowed amount, remove the box
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > maxOverlap)[0])))

	#Return the new bounding boxes as integers (easier to draw)
	return boxesOld[boxesNew].astype("int")



# Set the descriptor to look for people
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Read in the image from the command line
image = cv2.imread(sys.argv[1])

# Detects rectangles around 'people' in the image, does not draw them
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
	padding=(8, 8), scale=1.05)

# Deploy the rectangles to an array for easier calculations
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

# Apply NMS to the rectangles, using the given paramter (0-1)
boxesNew =  non_max_suppression(rects, float(sys.argv[2]))

# Drae the NMS'd rectangles
for (xA, yA, xB, yB) in boxesNew:
	cv2.rectangle(image, (xA, yA), (xB, yB), (randint(0, 255), randint(0, 255), randint(0, 255)), 5)

#Print out the number of 'people' found, not always accurate
print("Number Of People Found: ", len(boxesNew))

# Show the image with people highlighted
cv2.imshow("Found People", image)
cv2.waitKey(0)
