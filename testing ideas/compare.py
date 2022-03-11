import cv2
import numpy as np
import face_recognition
import mediapipe as mp


elonImage = face_recognition.load_image_file("basic test images/elon.jpeg")
elonImage = cv2.cvtColor(elonImage, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(elonImage)[0]
encodeElon = face_recognition.face_encodings(elonImage)[0]
cv2.rectangle(elonImage, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
img = cv2.imread('basic test images/elon.jpeg')
result = faceDetection.process(img)
if result.detections:
    for id, detection in enumerate(result.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = img.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(img, bbox, (255, 0, 255), 2)


cv2.imshow("face recognition", elonImage)
cv2.imshow('mediapipe', img)
cv2.waitKey(0)