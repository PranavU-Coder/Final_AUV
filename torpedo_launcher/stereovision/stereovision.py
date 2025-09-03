# ESSENTIAL IMPORTS

import cv2 as cv
import numpy as np
import time

# USING THE OTHER PYTHON SCRIPTS FOR THIS MAIN IMPORTANT FILE

import triangulation as tri
import calibration

# COUPLING WITH THE YOLO MODEL

from ultralytics import YOLO
model = YOLO('/home/pranav/Projects/Final_AUV/torpedo_launcher/bounding_boxes/runs/detect/train/weights/best.pt')

cap_right = cv.VideoCapture(0, cv.CAP_DSHOW)
cap_left = cv.VideoCapture(1, cv.CAP_DSHOW)

# SETUP PARAMETERS (SUBJECT TO CHANGE , IMPORTANT TO NOTE THAT ALL OF THESE VALUES ARE PUT IN BY ME FOR REFERENCE)

frame_rate = 120

# DISTANCE BETWEEN CAMERAS (IN CM)

b = 5.15

# FOCAL LENGTH OF THE CAMERA (IN MM)

f = 3.29  

# CAMERA'S FOV IN THE HORIZONTAL PLANE

alpha = 72.4  

target_classes = ['hole']

def get_detection_center(results, frame):

    centers = []
    boxes_info = []
    
    if results and len(results) > 0:
        for result in results:
            boxes = result.boxes
           
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    
                    cls_id = int(box.cls.cpu().numpy()[0])
                    name = result.names[cls_id]
                    
                    if name in target_classes:
                    
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        confidence = float(box.conf.cpu().numpy()[0])
                        
                        centers.append((center_x, center_y))
                        boxes_info.append({
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': name,
                            'confidence': confidence
                        })
                        
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'{name}: {confidence:.2f}', 
                                 (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 
                                 0.5, (255, 0, 255), 2)
    
    return centers, boxes_info

def main():

    while cap_right.isOpened() and cap_left.isOpened():
        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()
        
        if not success_right or not success_left:
            break
            
        start = time.time()
        
        # CALIBRATION TIME BABY

        frame_right, frame_left = calibration.undistortRectify(frame_left, frame_right)
        
        results_right = model(frame_right, verbose=False)
        results_left = model(frame_left, verbose=False)
        
        centers_right, boxes_right = get_detection_center(results_right, frame_right)
        centers_left, boxes_left = get_detection_center(results_left, frame_left)
        
        if not centers_right or not centers_left:
            cv.putText(frame_right, 'TRACKING LOST', (75, 50), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv.putText(frame_left, 'TRACKING LOST', (75, 50), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:

            center_right = centers_right[0] if centers_right else None
            center_left = centers_left[0] if centers_left else None
            
            if center_right is not None and center_left is not None:
                depth = tri.find_depth(center_right, center_left, 
                                     frame_right, frame_left, b, f, alpha)
                
                depth_text = f"Distance: {round(depth, 1)} cm"
                
                cv.putText(frame_right, depth_text, (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv.putText(frame_left, depth_text, (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                print(f"Depth: {round(depth, 1)} cm")
        
        end = time.time()
        total_time = end - start
        fps = 1 / total_time if total_time > 0 else 0
        
        cv.putText(frame_right, f'FPS: {int(fps)}', (20, 450), 
                  cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(frame_left, f'FPS: {int(fps)}', (20, 450), 
                  cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv.imshow("Right Frame", frame_right)
        cv.imshow("Left Frame", frame_left)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # DESTROY EVERYTHING

    cap_right.release()
    cap_left.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
