from unittest import result
import cv2
import numpy as np
import face_recognition
import os
import time
import mediapipe as mp

path = 'attendance images'
images = []
peopleNames = []
mylist = os.listdir(path)

print('Getting known faces')
for name in mylist:
    currentImage = cv2.imread(f'{path}/{name}')
    images.append(currentImage)
    peopleNames.append(os.path.splitext(name)[0])

def findEncoding(images):
    encodedList = []

    for img, name in zip(images, peopleNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        encoded = face_recognition.face_encodings(img)[0]
        encodedList.append(encoded)
        print(f"encoded {name}'s image")

    return encodedList

print('encoding faces...')
encodedList = findEncoding(images)
print('encoding completed')

print('now capturing video with camera')


cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.7)

while True:

    success, img = cap.read()

    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgS)
    frameFaces = []

    if result.detections:
        for id, detection in enumerate(result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.ymin * ih), int(bboxC.xmin * iw)+int(bboxC.width * iw), int(bboxC.ymin * ih)+int(bboxC.height * ih), int(bboxC.xmin * iw)
            frameFaces.append(bbox)
            # cv2.rectangle(img, (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)), (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 2)
    frameFacesEncoded = face_recognition.face_encodings(imgS, frameFaces)

    for encodedFace, faceLocation in zip(frameFacesEncoded, frameFaces):
        matches = face_recognition.compare_faces(encodedList, encodedFace)
        faceDistant = face_recognition.face_distance(encodedList, encodedFace)

        print(faceDistant)
        matchIndex = np.argmin(faceDistant)

        if matches[matchIndex]:
            name = peopleNames[matchIndex].upper()
            print(name)

            y1, x2, y2, x1 = faceLocation
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{name} {int(detection.score[0]*100)}%', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLocation
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "who?", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)