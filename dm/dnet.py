import cv2
import numpy as np
import imutils

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class DNet:
    def __init__(self,
                 prototxt,
                 model,
                 confidence = 0.2):
        self.prototxt = prototxt
        self.model = model
        self.confidence = confidence
        
        # load our serialized model from disk
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, frame):
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        objects = []

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence:
                idx = int(detections[0, 0, i, 1])
                objClass = "{}".format(CLASSES[idx])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                centerX = int((endX - startX) / 2 + startX)
                centerY = int((endY - startY) / 2 + startY)
                objects.append((objClass, confidence, centerX, centerY))

        return objects
