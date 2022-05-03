# import necessary libraries
import cv2
import cvzone
from time import sleep
from flask import Flask, Response, request, render_template
##########
#warm flask up.. set the capture image.. add a delay..
app = Flask(__name__)
cap = cv2.VideoCapture(0)
sleep(1)
##########


#necessary face detection vars and paths, not used
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))
###########

#necessary vars and paths for object detection.. important
classNames = []
classFile = 'coco.names' #change path on rpi
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #change path on rpi
weigthsPath = 'frozen_inference_graph.pb' #change path on rpi

thres = 0.45 #to detect objects
net = cv2.dnn_DetectionModel(weigthsPath,configPath) #net dnn detect model stuff
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
############

#main, with the welcome page route..
@app.route('/')
def index():
    # all this commenting stuff cuz stoopid redirect wont work with delay
    return render_template('indexWelcome.html')
    #return render_template('navBar.html')
@app.route('/SkyDash')
def skyDash():
    return render_template('index.html')
############



###################################################################################################
############################VOIDS FOR DETECTIONS, GRAY AND BASIC STREAM############################
###################################################################################################
#<\>
def find_obj(draw=True, objects=[]):
    while True:
        _, img = cap.read()
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
                        cvzone.cornerRect(img, box, l=50, t=15, rt=3,
                        colorR=(0, 0, 255), colorC=(0, 0, 0))
                        cv2.putText(img, className.upper(), (box[0]+10,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
                        cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
        #return objectInfo

        _, jpeg = cv2.imencode('.jpg', img)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def ObjDetMultiClass(draw=True):
    while True:
        objects = []
        _MultiDet, imgMultiDet = cap.read()
        classIds, confs, bbox = net.detect(imgMultiDet, confThreshold=0.5)
        #print(classIds, bbox)
        if len(objects) == 0: objects = classNames

        objectInfo = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if (draw):
                        #cv2.rectangle(imgMultiDet, box, color=(0,0,255), thickness=3)
                        cvzone.cornerRect(imgMultiDet, box, l=50, t=15, rt=3,
                        colorR=(0, 0, 255), colorC=(0, 0, 0))
                        cv2.putText(imgMultiDet, className.upper(), (box[0]+10,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
                        cv2.putText(imgMultiDet, str(round(confidence*100, 2)), (box[0]+200,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
        #return objectInfo

        _MultiDet, jpeg = cv2.imencode('.jpg', imgMultiDet)

        frameMultiDet = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frameMultiDet + b'\r\n\r\n')


def streamer(cap):
    while True:
        _Raw, imgRaw = cap.read()
        _Raw, jpegRaw = cv2.imencode('.jpg', imgRaw)

        #imgRaw = cv2.resize(imgRaw, (60, 40))

        #imgRaw = cv2.resize(imgRaw, (360, 240))
        frameRaw = jpegRaw.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frameRaw + b'\r\n\r\n')

def gray(cap):
    while True:
        _Gray, imgGray = cap.read()
        imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
        _Gray, jpegGray = cv2.imencode('.jpg', imgGray)

        frameGray = jpegGray.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frameGray + b'\r\n\r\n')
#</>
###################################################################################################
###################################################################################################
###################################################################################################



#############################################FLASK ROUTES##########################################
#<\>
@app.route('/objectDetect')
def detect():
    #multiclass = request.args.get('multiclass')
    obj = request.args.get('obj1')
    obj2 = request.args.get('obj2')
    obj3 = request.args.get('obj3')
    Draw = request.args.get('draw')
    Draw = bool(Draw)

    global img
    return Response(find_obj(draw=Draw, objects=[obj, obj2, obj3]),
                             mimetype= 'multipart/x-mixed-replace; boundary=frame')

@app.route('/MultiClassObjectDetect')
def MultiClassObjDet():
    DDraw = request.args.get('draw')
    DDraw = bool(DDraw)

    global imgMultiDet
    return Response(ObjDetMultiClass(draw=DDraw),
                             mimetype= 'multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def stream():
    global imgRaw

    return Response(streamer(cap),
                             mimetype= 'multipart/x-mixed-replace; boundary=frame')

@app.route('/gray')
def Gray():
    global imgGray
    return Response(gray(cap),
                             mimetype= 'multipart/x-mixed-replace; boundary=frame')
#</>
###################################################################################################


#running
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
