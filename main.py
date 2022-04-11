import cv2
import numpy as np
import sqlite3

# connecting to local databse
connection = sqlite3.connect("nutrition.db")
cursor = connection.cursor()

# Get frames of webcam
cap = cv2.VideoCapture(0)

# tranform resolution to multiple of 32 to allow Darknet CNN to process
WiHeTa = 320

# setting minimum confidence threshold and aggressiveness of model
confidenceThreshold = 0.7
nmsThreshold = 0.0005

# drag classes.names file in project folder
# read text file of class names
fruitFile = 'classes.names'
fruitNames = []
currentId = -1
currentInfo = []

# User Interface setup
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.4
color = (0, 0, 0)
thickness = 1
uiImg = cv2.imread("UI.jpg")
nutrition = False

# open text file and extract information
with open(fruitFile, 'rt') as f:
    fruitNames = f.read().rstrip('\n').split('\n')

# drag config file and unzipped weights file into the project folder
# can be downloaded from link in 'readme.md' file on git
modelConfiguration = 'yolov3_fruit.cfg'
modelWeights = 'yolov3_fruit_final.weights'

# connects to neural network
# Target CPU, dependencies for GPU leverage don't work :\
net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Renders a window with rectangle outlining detected object
# creates array for bounding box and class ids
# declares global variables
def findObjects(outputs, img):
    He, Wi, Ta = img.shape
    boundingBox = []
    classIds = []
    global currentId
    global currentInfo
    global uiImg
    global nutrition

    confidence_values = []

# if confidence of identified object is greater than the minimum threshold
# return name and confidence from neural network
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
    # pink
    for i in indices:
        box = boundingBox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{fruitNames[classIds[i]].upper()} {int(confidence_values[i]*100)}%',
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # pulls and displays information stored in 'nutrition' database
        # Data will only be printed on a change in class id
        # Information is displayed as text over the connected jpg
        if classIds[i] != currentId:
            uiImg = cv2.imread("UI.jpg")
            currentId = classIds[i]
            command = f"SELECT * FROM food WHERE id = '{currentId}'"
            print(cursor.execute(command).fetchall())
            currentInfo = cursor.execute(command).fetchall()
            name = cv2.putText(uiImg, currentInfo[0][1], (305, 75), font, 0.5, color, 1, cv2.LINE_AA)
            calories = cv2.putText(uiImg, currentInfo[0][14], (390, 128), font, 1, color, 3, cv2.LINE_AA)
            totalFat = cv2.putText(uiImg, currentInfo[0][2], (357, 175), font, fontScale, color, thickness, cv2.LINE_AA)
            saturatedFat = cv2.putText(uiImg, currentInfo[0][15], (357, 195), font, fontScale, color, thickness, cv2.LINE_AA)
            cholesterol = cv2.putText(uiImg, currentInfo[0][3], (357, 213), font, fontScale, color, thickness, cv2.LINE_AA)
            potassium = cv2.putText(uiImg, currentInfo[0][5], (357, 231), font, fontScale, color, thickness, cv2.LINE_AA)
            sodium = cv2.putText(uiImg, currentInfo[0][4], (357, 249), font, fontScale, color, thickness, cv2.LINE_AA)
            totalCarbohydrates = cv2.putText(uiImg, currentInfo[0][6], (357, 267), font, fontScale, color, thickness, cv2.LINE_AA)
            dietaryFibre = cv2.putText(uiImg, currentInfo[0][7], (357, 286), font, fontScale, color, thickness, cv2.LINE_AA)
            totalSugars = cv2.putText(uiImg, currentInfo[0][8], (357, 304), font, fontScale, color, thickness, cv2.LINE_AA)
            protein = cv2.putText(uiImg, currentInfo[0][9], (357, 340), font, fontScale, color, thickness, cv2.LINE_AA)
            vitaminD = cv2.putText(uiImg, currentInfo[0][13], (357, 369), font, fontScale, color, thickness, cv2.LINE_AA)
            vitaminC = cv2.putText(uiImg, currentInfo[0][10], (357, 387), font, fontScale, color, thickness, cv2.LINE_AA)
            iron = cv2.putText(uiImg, currentInfo[0][11], (357, 405), font, fontScale, color, thickness, cv2.LINE_AA)
            calcium = cv2.putText(uiImg, currentInfo[0][12], (357, 423), font, fontScale, color, thickness, cv2.LINE_AA)


while True:
    # give us image and verify success
    success, img = cap.read()

    # converts image to blob format
    blob = cv2.dnn.blobFromImage(img, 1/255, (WiHeTa, WiHeTa), [0, 0, 0], 1, crop=False)
    # set blob as input to network
    net.setInput(blob)

    # give names of necessary layers
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    # sends object names to neural network
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    # Shows video feed on screen
    # resizes window
    # attaches jpg to camera feed window, has to be same resolution
    cv2.namedWindow('Fruitify', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fruitify', 1920, 720)
    combinedImg = np.concatenate((img, uiImg), axis=1)
    cv2.imshow('Fruitify', combinedImg)

    # end application at 'q' button press
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        print("Application closed.")
        break
