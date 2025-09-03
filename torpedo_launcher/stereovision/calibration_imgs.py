import cv2 as cv

cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(2)

num = 0

while cap.isOpened():

    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    key = cv.waitKey(5)

    if key==27:
        break

    elif key == ord('s'):
        cv.imwrite('imgs/stereoLeft/imageL' + str(num) + '.png',img1)
        cv.imwrite('imgs/stereoRight/imageR' + str(num) + '.png',img2)
        print("images saved")
        num+=1

    cv.imshow('img 1',img1)
    cv.imshow('img 2',img2)

cap1.release()
cap2.release()

cv.destroyAllWindows()
