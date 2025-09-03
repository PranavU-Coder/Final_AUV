# ESSENTIAL IMPORTS

import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

# USING THE OTHER PYTHON SCRIPTS FOR THIS MAIN IMPORTANT FILE

import triangulation as tri
import calibration

# COUPLING WITH YOLO MODEL

import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap_right = cv.VideoCapture(0,cv.CAP_DSHOW)
cap_left = cv.VideoCapture(1,cv.CAP_DSHOW)

# SETUP PARAMETERS (SUBJECT TO CHANGE, PLEASE NOTE THESE VALUES ARE TEMPORARILY PUT UP BY ME FOR REFERENCING)

# MAXIMUM FPS IS 120

frame_rate = 120

# DISTANCE BETWEEN THE CAMERAS (IN CM)

b = 2

# FOCAL LENGTH OF THE CAMERAS (IN MM)

f = 4

# CAMERA'S FOV IN HORIZONTAL PLANE

alpha = 58

with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    
    while(cap_right.isOpened() and cap_left.isOpened()):
        
        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()
        
        # CALIBRATING IT BABY
        
        frame_right, frame_left = calibration.undistortRectify(frame_left, frame_right)
        
        if not success_right or not success_left:
            break
        else:
            start = time.time()
            
            # CONVERTING THE BGR TO RGB NOW
            
            frame_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)
            frame_left = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)
            
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)
            
            # ... AND GUESS WHAT
            
            frame_right = cv.cvtColor(frame_right, cv.COLOR_RGB2BGR)
            frame_left = cv.cvtColor(frame_left, cv.COLOR_RGB2BGR)
            
            # DEPTH ESTIMATION
            
            center_point_right = None
            center_point_left = None
            
            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_right.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_right = (boundBox[0] + boundBox[2]//2, boundBox[1] + boundBox[3]//2)
                    cv.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_left.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_left = (boundBox[0] + boundBox[2]//2, boundBox[1] + boundBox[3]//2)
                    cv.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if not results_right.detections or not results_left.detections:
                cv.putText(frame_right, 'TRACKING LOST', (75,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                cv.putText(frame_left, 'TRACKING LOST', (75,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            
            else:
                if center_point_right is not None and center_point_left is not None:
                    depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, b, f, alpha)
                    cv.putText(frame_right, "Distance : " + str(round(depth,1)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    cv.putText(frame_left, "Distance : " + str(round(depth,1)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    print("Depth : ", str(round(depth,1)))
            
            end = time.time()
            totalTime = end - start
            
            fps = 1/totalTime
            
            cv.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            cv.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            cv.imshow("right frame", frame_right)
            cv.imshow("left frame", frame_left)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

# DESTROY EVERYTHING

cap_right.release()
cap_left.release()
cv.destroyAllWindows()
