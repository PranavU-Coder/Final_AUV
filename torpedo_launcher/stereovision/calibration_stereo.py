import numpy as np
import cv2 as cv
import glob

chessBoardSize = (7,7)
frameSize = (640,480)

# THE TERMINATION CRITERIA NEEDED (AS GIVEN IN OPENCV'S DOCUMENTATION)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

objp = np.zero((chessBoardSize[0] * chessBoardSize[1] , 3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0],0:chessBoardSize[1]].T.reshape(-1,2)

# SUBJECT TO CHANGE

objp = objp * 18

print(objp)

objpoints = [] # ALL THE 3D POINTS IN REAL WORLD SPACE
imgpointsL = [] # 2D POINTS IN IMAGE PLANE
imgpointsR = [] # 2D POINTS IN IMAGE PLANE

imagesLeft = glob.glob('images/stereoLeft/imageL0.png')
imagesRight = glob.glob('images/stereoRight/imageR0.png')

for imgLeft , imgRight in zip (imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL , cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR , cv.COLOR_BGR2GRAY)

    # NOW FINDING THE CHESS BOARD CORNERS

    retL , cornersL = cv.findChessboardCorners(grayL , chessBoardSize , None)
    retR , cornersR = cv.findChessboardCorners(grayR , chessBoardSize , None)

    if retL and retR == True :
        
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL , (11,11) , (-1,-1),criteria)
        imgpointsL.append(cornersL)
        
        cornersR = cv.cornerSubPix(grayR, cornersR , (11,11) , (-1,-1),criteria)
        imgpointsR.append(cornersR)

        cv.drawChessboardCorners(imgL, chessBoardSize, cornersL, retL)
        cv.imshow('img left',imgL)
        cv.drawChessboardCorners(imgR, chessBoardSize, cornersR, retR)
        cv.imshow('img right',imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()

# ACTUAL CALIBRATION

retL , cameraMatrixL , distL , rvecsL , tvecsL = cv.calibrateCamera(objpoints, imgPointsL , frameSize , None , None)
heightL , widthL , channelsL = imgL.shape

newCameraMatrixL , roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL , distL , (widthL,heightL) , 1 , (widthL,heightL) , 0)

retR , cameraMatrixR , distR , rvecsR , tvecsR = cv.calibrateCamera(objpoints, imgPointsR , frameSize , None , None)
heightR , widthR , channelsR = imgR.shape
newCameraMatrixR , roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR , distR , (widthR,heightR) , 1 , (widthR,heightR) , 0)

# STEREOVISION CALIBRATION NOW

flags = 0
flags = cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints,
                                                                                                                                 imgpointsL,
                                                                                                                                 imgpointsR,
                                                                                                                                 newCameraMatrixL,
                                                                                                                                 distL,
                                                                                                                                 newCameraMatrixR,
                                                                                                                                 distR,
                                                                                                                                 frameSize,
                                                                                                                                 criteria=criteria_stereo,
                                                                                                                                 flags=flags)

# STEREO RECTIFICATION

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL,
                                                                           distL,
                                                                           newCameraMatrixR,
                                                                           distR,
                                                                           frameSize,
                                                                           rot,
                                                                           trans,
                                                                           alpha=rectifyScale)

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("saving parameters")

cv_file = cv.FileStorage('stereoMap.xml',cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()
