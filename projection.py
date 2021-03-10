"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 1 Projecting 3D Cube onto April Tag
"""
import numpy as np
import cv2
from numpy.core.fromnumeric import reshape

# Read the video file.
cap = cv2.VideoCapture('media/Tag1.mp4')
testudo = cv2.imread('media/testudo.png')

# Intrinsic camera parameters
K = [[1406.08415449821, 2.20679787308599, 1014.13643417416],
[0, 1417.99930662800, 566.347754321696], 
[0, 0, 1]]

K = np.reshape(K, (3, 3))

def cnts(image): # Use canny edge detection to define and display edges. Returns AR Tag x, y coordinates.

    k = 5
    blurr = cv2.GaussianBlur(image, (k, k), 0) # Blurr frame

    threshold_1 = 100 # Define Canny edge detection thresholds
    threshold_2 = 200 # Define Canny edge detection thresholds
    canny = cv2.Canny(blurr, threshold_1, threshold_2) # Call Canny edge detection on blurred image
    _, bianary = cv2.threshold(canny, 230, 255, 0) # Convert to bianary image

    contours, hierarchy = cv2.findContours(bianary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find all contours in image. cv2.RETR_TREE gives all hierarchy information, cv2.CHAIN_APPROX_SIMPLE stores only the minimum required points to define a contour.
        
    g_cnts = [] # Store only the contours with parents and children.
    g_hier = [] # Store hierarchy of contours with parents and children.
    
    for i in range(len(hierarchy)): # Filter out contours with no parent to isolate ar tag.
        for j in range(len(hierarchy[i])):
            if hierarchy[i][j][3] != -1 and hierarchy[i][j][2] != -1:
                g_cnts.append(contours[j])
                g_hier.append(hierarchy[i][j])

    area = [] # Store area of filtered contours with parents and children
    for i in range(len(g_cnts)):
        area.append(cv2.contourArea(g_cnts[i]))

    final_cnts = [] # Store AR Tag contours
    for i in range(len(g_cnts)):
        if area[i] < max(area):
            final_cnts.append(g_cnts[i])  

    x_tag =[] # Store x coordinates of AR Tag contour
    y_tag =[] # Store y coordinates of AR Tag contour
    for i in range(len(final_cnts)):
        for j in range(len(final_cnts[i])):
            x_tag.append(final_cnts[i][j][0][0])
            y_tag.append(final_cnts[i][j][0][1]) 

    # cv2.drawContours(image, final_cnts, -1, (0, 255, 0), 3)

    return x_tag, y_tag

def decode(video_frame, x_tag, y_tag): # Decode AR Tag for orientation.
    
    dec_img = np.ones_like(video_frame) # Creates blank image to map AR Tag to.
    dec_img = dec_img*video_frame # Map frame to blank image.
    try:
        dec_img = dec_img[min(y_tag):max(y_tag), min(x_tag):max(x_tag)] # Crop to AR Tag.
    except ValueError:
        pass

    img = cv2.resize(dec_img,(800,800)) # Resize image to be standard 800x800 before segmenting and decoding.

    segments = segm(img) # Segment the isolated AR Tag
    tag_id, key = id_chk(segments) # Decode AR Tag. Return Tag ID and Orientation key.
    print('AR-Tag ID: ' + tag_id + ' ' + key)

    # cv2.imshow('dec_tag', dec_img) # Shows isolated AR Tag

    return key

def segm(img):# Segment the image into 8x8 grid for a total of 16 segments.
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

def id_chk(segments): # Checks the values of the outer corners. If 1 then ID the tag starting from the inner opposing corner proceeding clockwise.
    global last_tag_id
    idx = [0, 3, 5, 6, 9, 10, 12, 15]
    id = []
    tag_id = ''
    key = ''
    for i in range(len(idx)):
        if np.mean(segments[idx[i]]) > 127:
            id.append(1)
        else:
            id.append(0)

    if id[idx.index(0)] == 1: # Check upper left corner
        # Start at [10], [9], [5], [6]
        tag_id = str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)])
        key = 'TL'
        last_tag_id = tag_id

    elif id[idx.index(3)] == 1: # Check upper right corner
        # Start at [9], [5], [6], [10]
        tag_id = str(id[idx.index(9)]) + str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)])
        key = 'TR'
        last_tag_id = tag_id

    elif id[idx.index(12)] == 1: # Check lower left corner
        # Start at [6], [10], [9], [5]
        tag_id = str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)]) + str(id[idx.index(5)])
        key = 'BL'
        last_tag_id = tag_id

    elif id[idx.index(15)] == 1: # Check lower right corner
        # Start at [5], [6], [10], [9]
        tag_id = str(id[idx.index(5)]) + str(id[idx.index(6)]) + str(id[idx.index(10)]) + str(id[idx.index(9)])
        key = 'BR'
        last_tag_id = tag_id

    else: # If cannot calculate ID or key, use previous itterations values.
        key = last_key
        tag_id = last_tag_id

    return tag_id, key

def tag_corners(x_tag, y_tag): # Returns the 4 corners of the AR Tag and orientation.

    try:
        x_11, y_11 = min(x_tag), min(y_tag) # TL
        x_22, y_22 = max(x_tag), min(y_tag) # TR
        x_33, y_33 = min(x_tag), max(y_tag) # BL
        x_44, y_44 = max(x_tag), max(y_tag) # BR

    except ValueError: # If cannot calculate coordinates, use previous itterations values.
            x_tag = last_x_tag 
            y_tag = last_y_tag
    else:
        x_tag = [x_11, x_22, x_33, x_44]
        y_tag = [y_11, y_22, y_33, y_44]

    return x_tag, y_tag

def img_corners(img, key): # Returns the 4 corners of the image to be superimposed.
    if key == 'BR':
        pass
    elif key == 'BL':
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif key == 'TR':
        img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    else:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width, _ = img.shape
    x_1, y_1 = 0, 0 # TL
    x_2, y_2 = width, 0 # TR
    x_3, y_3 = 0, height # BL
    x_4, y_4 = width, height # BR

    x_img = [x_1, x_2, x_3, x_4]
    y_img = [y_1, y_2, y_3, y_4]

    return x_img, y_img, img
def cube_coordinates(cube):
    x_1, y_1 = cube[0][0], cube[0][1] # TL
    x_2, y_2 = cube[1][0], cube[1][1] # TR
    x_3, y_3 = cube[2][0], cube[2][1] # BL
    x_4, y_4 = cube[3][0], cube[3][1] # BR

    x_img = [x_1, x_2, x_3, x_4]
    y_img = [y_1, y_2, y_3, y_4]

def homography(x_tag_corners, y_tag_corners, x_img, y_img): # Compute the homography between AR Tag and image.

    # Get tag corners and orientation from frame. x_11 being top left.
    xp = x_tag_corners
    yp = y_tag_corners

    # Get image corners
    x = x_img
    y = y_img

    # A matrix construction
    try:
        A = np.array([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]], 
        [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
        [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
        [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
        [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
        [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
        [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
        [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])
    
    except IndexError:
        H = last_H

    else:
        u, s, v = np.linalg.svd(A)

        H = v[-1]
        H = np.reshape(H, (3, 3))

    return H

def proj(video_frame, image, h): # Project image onto the detected AR Tag

    width, height, _ = image.shape

    # Compute lambda
    lam = (np.linalg.norm(np.dot(np.linalg.inv(K),h[0])) + np.linalg.norm(np.dot(np.linalg.inv(K),h[1]))) / 2
    lam = lam**-1

    # Compute B_hat = lam*k^-1*H
    B_hat = lam * np.dot(np.linalg.inv(K), h)

    B = lam * B_hat

    R = [[B[0][0], B[0][1], B[0][0] * B[1][0]], 
        [B[1][0], B[1][1], B[1][0] * B[1][1]], 
        [B[2][0], B[2][1], B[2][0] * B[2][1]]]
    
    R = np.reshape(R, (3, 3))

    t = [B[0][2], B[1][2], B[2][2]]

    t = np.reshape(t, (3,1))

    points, _ = cv2.projectPoints(cube, R, t, K, np.zeros(4))

    points = np.int32(points).reshape(-1, 2)
    video_frame = cv2.drawContours(video_frame, [points[:4]], -1, (0, 255, 0), 2)
    video_frame = cv2.drawContours(video_frame, [points[4:]], -1, (255, 0, 0), 2)
    for i, j in zip(range(4), range(4, 8)):
        video_frame = cv2.line(video_frame, tuple(points[i]), tuple(points[j]), (0, 0, 255), 2)
    

    # Compute projection matrix P.
    # P = np.dot(K, B)

    # # # P * image?
    # # Set video frame tag pixels equal to the image pixels by multiplyinh by homography
    # for j in range(width): 
    #     for i in range(height): 
    #         tag_vec = np.dot(P,[i,j,1]) 
            
    #         if tag_vec[-1] != 1: # If last value in vector is not 1, divide teh vector by the last value to make it 1
    #             tag_vec = tag_vec/tag_vec[-1] 
    #             try:
    #                 video_frame[int(tag_vec[1])][int(tag_vec[0])] = image[j][i] # Set the frame coordinate equal to the corisponting image coordinate value
    #             except IndexError:
    #                 pass
    #         else:
    #             video_frame[int(tag_vec[1])][int(tag_vec[0])] = image[j][i]

# Variables used to store previous frames information. If current frame returns nothing for any of the variables, use its coresponding previous value instead.
last_key = '' # Previous Tag orientation key.
last_x_tag = [] # Previous Tag x coordinates.
last_y_tag = [] # Previous Tag y coordinates.
last_tag_id = '' # Previous Tag ID.
last_H = [] # Previous homography
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('projection.mp4', fourcc, 20.0, (1920, 1080))

while True:
        
    ret, frame = cap.read()

    if ret == False:
        break

    cube = np.float32([[0, 0, 0], [0, 400, 0], [400, 400, 0], [400, 0, 0], 
    [0, 0, -400], [0, 400, -400], [400, 400, -400], [400, 0, -400]])

    x_tag, y_tag = cnts(frame) # Use canny edge detection to define edges and detect AR Tag Corners

    key = decode(frame, x_tag, y_tag) # Decode frame and return key value describing the orientation of the AR Tag.
  
    last_key = key
    last_x_tag = x_tag
    last_y_tag = y_tag

    x_img, y_img, fliped_img = img_corners(testudo, key) # Return the image corner coordinates and reoriented image.
    x_tag_corners, y_tag_corners = tag_corners(x_tag, y_tag) # Return the AR Tag coordinates
    h = homography(x_tag_corners, y_tag_corners, x_img, y_img) # Compute the homography between AR Tag and Flipped Image

    last_H = h

    proj(frame, fliped_img, h) # Project the Flipped Image over the AR Tag.

    cv2.imshow("Video Feed", frame) # Show the projection
    out.write(frame) # Save output as video.

    # Condition to break the while loop
    i = cv2.waitKey(50)
    if i == 27:
        break    

# Wait for ESC key to be pressed before releasing capture method and closing windows
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

"""
Isolate tag in video

Get tag coordinates and orientation 
and set origin of world cordinates to be aligned with
the orientation of the AR tag. 

Get Testudo image coordinates.

Compute homography.

    # Part 2.b
    Compute B matrix, 
    if determinant of B is negative, 
    multiply by -1 to assume object is located infront of the camera.
    col of B are denoted as b1 b2 b3

    Compute lam = ((magnitude(K_inv * h1) + magnitude(K_inv * h2)) / 2)^-1

    Compute rotation matrix R: r1 = lam * b1 r2 = lam * b2 r3 = r1xr2 
    Compute transformation vector: t = lam * b3

Construct projection matrix P = K[R | t] for part 2.a P

Multiply P by the testudo image and overlay with the AR tag video.
"""