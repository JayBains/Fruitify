import cv2
import numpy as np
import sqlite3

connection = sqlite3.connect("nutrition.db")
cursor = connection.cursor()

# Get frames of webcam
cap = cv2.VideoCapture(0)
WiHeTa = 320
confidenceThreshold = 0.7
nmsThreshold = 0.0005

fruitFile = 'classes.names'
fruitNames = []
currentId = -1
currentInfo = []

# open text file and extract information
with open(fruitFile, 'rt') as f:
    fruitNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3_fruit.cfg'
modelWeights = 'yolov3_fruit_final.weights'

net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    He, Wi, Ta = img.shape
    boundingBox = []
    classIds = []
    global currentId
    global currentInfo

    confidence_values = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w, h = int(det[2]*Wi), int(det[3]*He)
                x, y = int((det[0]*Wi) - w/2), int((det[1]*He) - h/2)
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confidence_values.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boundingBox, confidence_values, confidenceThreshold, nmsThreshold)

    # creates a bounding box with name and confidence percentage on identified objects
    for i in indices:
        box = boundingBox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{fruitNames[classIds[i]].upper()} {int(confidence_values[i]*100)}%',
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.putText(img, f'{currentInfo}',
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Displays information stored in 'nutrition' database
        # Data will only be printed on a change in class id, so variable current id is used as comparison
        if classIds[i] != currentId:
            currentId = classIds[i]
            command = f"SELECT * FROM food WHERE id = '{currentId}'"
            print(cursor.execute(command).fetchall())
            currentInfo = cursor.execute(command).fetchall()




while True:
    # give us image and verify success
    success, img = cap.read()

    # converts image to blob format
    blob = cv2.dnn.blobFromImage(img, 1/255, (WiHeTa, WiHeTa), [0, 0, 0], 1, crop=False)
    # set blob as input to network
    net.setInput(blob)

    # give all names of layers
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    # Shows video feed on screen
    cv2.imshow('Image', img)

    # close application at 'q' button press
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        print("Application closed.")
        break
