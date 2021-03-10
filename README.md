ENPM 673 - Perception for Autonomous Robots: 

Project 1 - AR Tag Detection, Decoding, Tracking/Projection and Homography 

Shon Cortes


detection.py:

    Packages Used:

        1. import numpy
        2. import cv2 
        3. import matplotlib
        4. import matplotlib.pyplot as plt
        5. import scipy
        6. from scipy.fft import  fft, fftshift, ifft, ifftshift

    Run the program:

            1. Run the program using your prefered method (terminal...)

            Example in Terminal:

                python3 detection.py
            
            Output will be a plot of 4 images showing the original image, fft image, low pass filter applied to fftshifted image, and the final result after inverse fft.
            
        Program Summary:
            
            The input image is of an April Tag on a dark floor with a repetitive textured pattern. The detect method takes the image as an input and first converts it to a grayscale image. The gray scale image is then thresholded to return a binary image. The binary image is blurred and the fft is performed on the binary image. The fft converts the image into the frequency domain with the low frequency components in the corners. The converted image then has the low frequency components shifted to the center. A circular low pass filter is then applied to filter out the high frequency textured background from the ground. This filtered image can then have the low frequency components shifted back to the corners. After shifting back to the corners, inverse fft can is performed to return the isolated AR Tag image with the background filtered out. The final output shows the major steps in a plot.
            decode.py:

    Packages Used:

        1. import numpy
        2. import cv2 

    Run the program:

            1. Run the program using your prefered method (terminal...)
                If alternate images need to be tested, change the file path on line 9:

                    - img = cv2.imread('media/ref_marker.png')

                Additional test markers were provided in: media/test_images. Choose from ref_marker2.png through ref_marker5.png

            Example in Terminal:

                python3 decode.py
            
            Output will display final decoded AR Tag with ID and corner coordinates overlayed.
            
        Program Summary:
            
            The input image is of an April Tag. It is first segmented into an 8x8 grid and the outer 2 most squares of the grid are removed to get a final 4x4 grid and a total of 16 segments. These segments are are then thresholded to determine if they are white or black. The corners are then checked to determine the orientation of the Tag. The orientation is used to then determine the AR Tag ID. The tag ID and its edge coordinates are then displayed on the image.
            
            
tracking.py:

    Packages Used:

        1. import numpy
        2. import imutils
        3. import cv2 

    Run the program:

            1. Run the program using your prefered method (terminal...)
                If alternate videos need to be tested, change the file path on line 11:

                    - cap = cv2.VideoCapture('media/Tag1.mp4')

                Additional videos were provided in: media/ folder. Choose from Tag0.mp4 through Tag2.mp4.

            Example in Terminal:

                python3 tracking.py
            
            Output will display the final Testudo image projected onto the AR Tag in each frame.
            
        Program Summary:

            The input image is of an April Tag. It is first segmented into an 8x8 grid for a total of 16 segments. These segments are then thresholded to determine if they are white or black. The corners are then checked to determine the orientation of the Tag. The orientation is used to orientate our testudo image (or any image provided) before computing the homography between the AR Tag coordinates and the provided image. Using the homography, the provided image is projected onto the AR Tag in the frame.

            x_11, y_11 = min(x_tag), y_tag[x_tag.index(min(x_tag))] # TL
        x_22, y_22 = x_tag[y_tag.index(min(y_tag))], min(y_tag) # TR
        x_33, y_33 = x_tag[y_tag.index(max(y_tag))], max(y_tag) # BL
        x_44, y_44 = max(x_tag), y_tag[x_tag.index(max(x_tag))] # BR