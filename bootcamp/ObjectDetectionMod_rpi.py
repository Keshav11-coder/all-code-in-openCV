import cv2
import cvzone
thres = 0.45 #to detect objects

#img = cv2.imread('cat3.jpg')


classNames = []
classFile = '/home/pi/recources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/home/pi/recources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthsPath = '/home/pi/recources/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weigthsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def find_obj(img,draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    #print(classIds, bbox)
    if len(objects) == 0: objects = classNames

    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    #cv2.rectangle(img, box, color=(0,0,255), thickness=3)
                    cvzone.cornerRect(img, box, l=30, t=10, rt=2,
                    colorR=(0, 0, 255), colorC=(0, 0, 0))
                    cv2.putText(img, className.upper(), (box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
    return img, objectInfo

def track_obj():
  print("vals")


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        _, img = cap.read()
        result, objectInfo = find_obj(img, objects=['person'])
        print(objectInfo)
        cv2.imshow("output", img)
        cv2.waitKey(1)