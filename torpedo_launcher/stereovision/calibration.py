import numpy as np
import cv2 as cv

cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml',cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

def undistortRectify(frameL , frameR):

    undistortedL = cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
    undistortedR = cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)
    
    return undistortedL, undistortedR
