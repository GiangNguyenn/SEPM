import cv2
import numpy as np
import face_recognition

elonImage = face_recognition.load_image_file("basic test images/elon.jpeg")
elonImage = cv2.cvtColor(elonImage, cv2.COLOR_BGR2RGB)
elonTest = face_recognition.load_image_file("basic test images/bill.jpeg")
elonTest = cv2.cvtColor(elonTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(elonImage)[0]
encodeElon = face_recognition.face_encodings(elonImage)[0]
cv2.rectangle(elonImage, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(elonTest)[0]
encodeTest = face_recognition.face_encodings(elonTest)[0]
cv2.rectangle(elonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(result, faceDis)
cv2.putText(elonTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow("elon", elonImage)
cv2.imshow("test", elonTest)
cv2.waitKey(0)