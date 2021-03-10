"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 1 Decode April Tag
"""
import numpy as np
import cv2

img = cv2.imread('media/ref_marker.png')
# img = cv2.imread('test_tag.png') # Test Tag image
img = cv2.resize(img,(800,800))

def debug(segments):# Debugging: Display all segmented portions in the image.
    cv2.imshow('0', segments[0])
    cv2.imshow('3', segments[3])
    cv2.imshow('5', segments[5])
    cv2.imshow('6', segments[6])
    cv2.imshow('9', segments[9])
    cv2.imshow('10', segments[10])
    cv2.imshow('12', segments[12])
    cv2.imshow('15', segments[15])

def id_chk(segments): # Checks the values of the outer corners. If 1 then ID the tag starting from the inner opposing corner proceeding clockwise.
    idx = [0, 3, 5, 6, 9, 10, 12, 15]
    id = []
    tag_id = ''
    for i in range(len(idx)):
        if np.mean(segments[idx[i]]) > 127:
            id.append(1)
        else:
            id.append(0)

    if id[idx.index(0)] == 1: # Check upper left corner
        # Start at [10], [9], [5], [6]
        tag_id = str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)])

    elif id[idx.index(3)] == 1: # Check upper right corner
        # Start at [9], [5], [6], [10]
        tag_id = str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)])

    elif id[idx.index(12)] == 1: # Check lower left corner
        # Start at [6], [10], [9], [5]
        tag_id = str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)])

    elif id[idx.index(15)] == 1: # Check lower right corner
        # Start at [5], [6], [10], [9]
        tag_id = str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)])
        
    return tag_id

def tag_def(img):# Segment the image into 8x8 grid for a total of 16 segments.
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = grey.shape
    x_grid = int(width/8)
    y_grid = int(height/8)

    segments = []

    for y in range(2 * y_grid, height - (2 * y_grid), y_grid): 

        for x in range(2 * x_grid, width - (2 * x_grid), x_grid):
            block = img[y : y + y_grid, x : x + x_grid]
            segments.append(block)

    return segments

def img_corners(img): # Returns the 4 corners of the image to be superimposed.
    height, width, _ = img.shape
    x_1, y_1 = 0, 0
    x_2, y_2 = width, 0
    x_3, y_3 = 0, height
    x_4, y_4 = width, height
    return x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4

segments = tag_def(img) # Call function to segment image.

# Check Blocks for their bianary value and return the image tag ID
tag_id = id_chk(segments)
x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = img_corners(img)

print('AR-Tag ID: ' + tag_id)

# Overlay Tag ID and Corner coordinates onto image.
cv2.putText(img, 'AR-Tag ID: ' + tag_id, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv2.putText(img, 'Corners: ' + '(' + str(x_1) + ', ' + str(y_1) +') ' +  '(' + str(x_2) + ', ' + str(y_2) +') '+ '(' + str(x_3) + ', ' + str(y_3) +') '+ '(' + str(x_4) + ', ' + str(y_4) +') ', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

cv2.imshow('AR-Tag Decoded', img) # Show final decoded AR Tag with ID and corner coordinates overlayed.

# Wait for ESC key to be pressed before releasing capture method and closing windows
cv2.waitKey(0)
cv2.destroyAllWindows()
