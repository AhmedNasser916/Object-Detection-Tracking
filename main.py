from cv2 import VideoCapture
from cv2 import destroyAllWindows
from cv2 import destroyWindow
from cv2 import waitKey
from cv2 import imshow
from cv2 import CAP_PROP_FPS
from cv2 import CAP_PROP_FRAME_HEIGHT
from cv2 import CAP_PROP_FRAME_WIDTH
from cv2 import createBackgroundSubtractorMOG2
from cv2 import threshold
from cv2 import THRESH_BINARY
from cv2 import THRESH_BINARY_INV
from cv2 import selectROI
from cv2 import findContours
import cv2

from tracker import *


def ini(vidoe):
    ret, frame = vidoe.read()
    frame = cv2.resize(frame, (1280, 720))
    imshow('Frame', frame)
    roi = selectROI('Frame', frame)
    destroyWindow('Frame')
    return roi


tracker = EuclideanDistTracker()
video = VideoCapture('highway_2.mp4')
# fps=CAP_PROP_FPS(video)
# hight=CAP_PROP_FRAME_HEIGHT(video)
# width=CAP_PROP_FRAME_WIDTH(video)
object_detector = createBackgroundSubtractorMOG2(history=100, varThreshold=40)

roiw = ini(video)

print(roiw)
while video.isOpened():
    check, frams = video.read()
    if check:
        frams = cv2.resize(frams, (1280, 720))
        roi = frams[int(roiw[1]):int((roiw[1] + roiw[3])), int(roiw[0]):int((roiw[0] + roiw[2]))]
        # roi = frams[340: 720, 500: 800]
        mask = object_detector.apply(roi)
        imshow("maskbefor", mask)
        _, mask = threshold(mask, 254, 255, THRESH_BINARY)

        contours, _ = findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:

            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 1)

    imshow("frams", frams)
    imshow("maskafter", mask)
    imshow("msss", roi)
    key = waitKey(30)
    if key == ord('q'):
        break

video.release()
destroyAllWindows()
