import cv2
import numpy as np
#detecting face exists or not in the image using harcascade classifier
face_classifier = cv2.CascadeClassifier('C:/Users/MY LENOVO/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# extracting features of face of the image and crop the image
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

#configure camera to capture a frame through the camera lens
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1 # resize the image,convert the image to gray scale
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# saving the image in a specific path
        file_name_path = 'C:/Users/MY LENOVO/Downloads/OpenCV-master/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face) #path,image
# the folder should not have any other image apart from the image
# count the images that are in the folder with putText
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass
# 13 -> ASIIC code for breaking the program and closing the camera 
# count = 100 represents how many samples are collected by the code from the camera
    if cv2.waitKey(1)==13 or count==100:
        break
cap.release()
cv2.destroyAllWindows()
print('Sample Collection Complete!!!')