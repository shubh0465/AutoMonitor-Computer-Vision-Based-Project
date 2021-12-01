import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path= 'images'
image=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for i in myList:
    img=cv2.imread(f'{path}/{i}')
    image.append(img)
    classNames.append(os.path.splitext(i)[0])
print(classNames)

def findEncodings(image):
    encodeList=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('presented.csv','r+') as f:
        myDatalist= f.readlines()
        nameList=[]
        for line in myDatalist:
            entry= line.split(',')
            nameList.append(entry[0])
        # if name not in nameList:
        #     now=datetime.now()
        #     dtstring= now.strftime('%H:%M:%S')
        #     f.writelines(f'\n{name}, {dtstring}')
        now = datetime.now()
        dtstring = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name}, {dtstring}')

# markAttendence('SHUBHAM')

encodeListForKnownFaces=  findEncodings(image)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    isTrue, img= cap.read()
    vid= cv2.resize(img,(0,0),None,0.25,0.25)
    vid= cv2.cvtColor(vid,cv2.COLOR_BGR2RGB)

    facesInCurrentFrame= face_recognition.face_locations(vid)
    encodeCurrentFrame = face_recognition.face_encodings(vid,facesInCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame,facesInCurrentFrame):
        matches= face_recognition.compare_faces(encodeListForKnownFaces,encodeFace)
        faceDistance= face_recognition.face_distance(encodeListForKnownFaces,encodeFace)
        print(faceDistance)
        print(matches)
        matchIndex= np.argmin(faceDistance)
        print(matchIndex)
        print(matches[matchIndex])
        print(faceDistance[matchIndex])
        if matches[matchIndex]:
            name= classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
        else:
            name = "UNKNOWN"
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('WebCam', img)
    cv2.waitKey(1)
# img=face_recognition.load_image_file("images/shubham.jpg")
# img=cv2.resize(img,(500,300))
# img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# imgtest=face_recognition.load_image_file("images/shubham_testing.jpg")
# imgtest=cv2.resize(imgtest,(480,680))
# imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
#
# faceloc= face_recognition.face_locations(img)[0]
# encodeFace= face_recognition.face_encodings(img)[0]
# cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,0,255),2)
#
# faceloctest= face_recognition.face_locations(imgtest)[0]
# encodetest= face_recognition.face_encodings(imgtest)[0]
# cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(0,0,255),2)
#
# result=face_recognition.compare_faces([encodeFace],encodetest)
# facedis = face_recognition.face_distance([encodeFace],encodetest)
# print(result,facedis)
# cv2.putText(imgtest,f'{result}{round(facedis[0],2)}',(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
# cv2.imshow("shubham",img)
# cv2.imshow("shubham testing",imgtest)
# cv2.waitKey(0)