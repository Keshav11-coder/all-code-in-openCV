import cv2
import numpy as np

w, h = 360, 275
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0
udPrvError = 0

#   Object Detection Algorithm in xml format
face_cascade = cv2.CascadeClassifier('/home/pi/haarcascades/haarcascade_frontalface_default.xml')

#video feed from the webcam
video = cv2.VideoCapture(0)

#   function to find da face
def findFace(video):
    myFaceListC = []
    myFaceListArea = []

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))

        return video, [myFaceListC[i], myFaceListArea[i]]
    else:
        return video, [[0, 0], 0]

#function to follow the face (a buncha errs here*)
def trackFace(info, w, pid, pError, udPrvError):
    area = info[1]

    x, y = info[0]
    fb = 0
    error = x - w // 2
    udError = y - h // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    ud = pid[0] * udError + pid[1] * (udError - udPrvError)
    speed = int(np.clip(speed, -100, 100))
    ud = int(np.clip(ud, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0

    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        ud = 0
        error = 0
    lr = str("0")
    udM = str("0")
    fbM = str("0")
    if speed < 50:
        lr = str("left")
    if speed > 50:
        lr = str("right")
    if ud < 50:
        udM = str("up")
    if ud > 50:
        udM = str("down")
    if fb == -20:
        fbM = str("backwards!")
    if fb == 20:
        fbM = str("forward!")
    print(lr, udM, fbM)
    return error


while True:
    # video getting read from frames
    (_, img) = video.read()

    faces = face_cascade.detectMultiScale(img,
                                          scaleFactor=1.1, minNeighbors=5)

    video, info = findFace(video)
    pError = trackFace(info, w, pid, pError, udPrvError)
    udPrvError = trackFace(info, w, pid, pError, udPrvError)

    cv2.imshow('results', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

#credit to mr akash for base code, modified it a little with extra tweeks and features
#credit to stackoverflow, for helping me solve errs*
#credit to "miss" and "mister(not really tho)" for gpio lecture and blink code using pi zero W
#credit to murtazas workshop, cv zone guy, inspiration to do this dumb and crazy(but amazing and GREAT) stuff
#cya'll folks, have a great life! don't play outside 