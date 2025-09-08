import cv2
from picamera2 import Picamera2, Preview

camL = Picamera2(0)
camR = Picamera2(1)

config_L = camL.create_preview_configuration({"format": "RGB888"})
config_R = camR.create_preview_configuration({"format": "RGB888"})

camL.configure(config_L)
camR.configure(config_R)

camL.start()
camR.start()

num = 0

try:
    while True:
        
        imgL = camL.capture_array()
        imgR = camR.capture_array()

        
        cv2.imshow("Left Camera", imgL)
        cv2.imshow("Right Camera", imgR)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:        
            break
        elif key == ord("s"):
            
            cv2.imwrite(f"imgs/stereoLeft/imageL{num}.png", imgL)
            cv2.imwrite(f"imgs/stereoRight/imageR{num}.png", imgR)
            print(f"Saved pair #{num}")
            num += 1

finally:

    camL.stop()
    camR.stop()
    cv2.destroyAllWindows()
