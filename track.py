import argparse
import time

import cv2
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('-v', '--video', help='Path to the (optional) video file.')
arguments = vars(argparser.parse_args())

blue_lower = np.array([100, 67, 0], dtype='uint8')
blue_upper = np.array([255, 128, 50], dtype='uint8')

camera = cv2.VideoCapture(arguments['video'])

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    blue = cv2.inRange(frame, blue_lower, blue_upper)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)

    (contours, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        rectangle = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(frame, [rectangle], -1, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    cv2.imshow('Binary', blue)

    time.sleep(0.025)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
