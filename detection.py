"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 1 Detection of April Tag
"""

import numpy as np
import cv2
import matplotlib 
import matplotlib.pyplot as plt
import scipy
from scipy.fft import  fft, fftshift, ifft, ifftshift

def detection(image): # Perform fft. Shift low frequency components to the center. Create a circular mask to filter out high frequeny components. Inverse fftshift, inverse fft to display the final filtered image showing only the AR Tag.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Kernel
    k = 5

    # Blurr first
    blurr = cv2.GaussianBlur(thresh, (k, k), 0)

    # FFT of frame
    f = cv2.dft(np.float32(blurr), flags = cv2.DFT_COMPLEX_OUTPUT)

    # Shift low frequency content at corners to the center
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum of the fft shifted frames
    mag_spect = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))

    # Define circular mask high pass filter
    height, width = gray.shape
    center_x = width/2
    center_y = height/2
    mask = np.zeros((height, width, 2), np.uint8)
    r = 200

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if (j-center_x)**2 + (i - center_y)**2 <= r**2:
                mask[i][j] = 1

    # Apply high pass filter to reduce noise by filtering out low frequency content
    mfshift = fshift*mask

    # Compute the magnitude spectrum of the fft shifted frames with high pass filter
    f_mag_spect = 20*np.log(cv2.magnitude(mfshift[:,:,0], mfshift[:,:,1]))

    # Apply inverse FFT on the high pass filtered frame
    ifft = np.fft.ifftshift(mfshift)
    filter_frame = cv2.idft(ifft)

    if_mag_spect = cv2.magnitude(filter_frame[:,:,0], filter_frame[:,:,1])

    return mag_spect, f_mag_spect, if_mag_spect

def plots(mag_spect, f_mag_spect, if_mag_spect): # Plot the images.
    plt.subplot(2,2,1),plt.imshow(frame, cmap= 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(mag_spect,  cmap= 'gray')
    plt.title('FFT Shifted'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(f_mag_spect,  cmap= 'gray')
    plt.title('FFT Shifted + Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(if_mag_spect,  cmap= 'gray')
    plt.title('IFFT + Mask'), plt.xticks([]), plt.yticks([])

    plt.show()

# Read the image file in gray scale
frame = cv2.imread('media/Tag1.png')

mag_spect, f_mag_spect, if_mag_spect = detection(frame)

plots(mag_spect, f_mag_spect, if_mag_spect)
