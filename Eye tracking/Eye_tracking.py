import cv2
import numpy as np

cap = cv2.VideoCapture("Pranav_DarkBrown_Vertical_01.mov")
value_list = list()
while True:
    ret, frame = cap.read() # read the frame

    if ret is False:
        break
    roi = cv2.blur(frame, (15,15)) #blurring image
    gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) # Convert to grayscale
    _, threshold = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV) # defining threshold
    edges = cv2.Canny(threshold,100,200) #Using canny edge

    contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # finding contours of threshold
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt) #finding connecting points to draw rectangle of pupil
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("Roi", roi)
    cv2.imshow("Edges", edges)
    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()