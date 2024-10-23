import numpy as np
import cv2

from FaceAlignment import FaceAlignment


class FacialExpressionRecog:

    expression = ["Angry",
                  "Disgust",
                  "Fearful",
                  "Happy",
                  "Neutral",
                  "Sad",
                  "Surprised"]

    def __init__(self,
                 modelPath,
                 backendId=0,
                 targetId=0):

        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._align_model = FaceAlignment()

        self._inputNames = 'data'
        self._outputNames = ['label']
        self._inputSize = [112, 112]
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._expression = ["Angry",
                            "Disgust",
                            "Fearful",
                            "Happy",
                            "Neutral",
                            "Sad",
                            "Surprised"]

    def infer(self, image, bbox=None):
        if bbox is not None:
            img = self._align_model.get_align_image(image,
                                                    bbox[4:].reshape(-1, 2))
        else:
            img = image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32, copy=False) / 255.0
        img -= self._mean
        img /= self._std

        inputBlob = cv2.dnn.blobFromImage(img)

        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward(self._outputNames)

        results = np.argmax(outputBlob[0], axis=1).astype(np.uint8)
        return FacialExpressionRecog.expression[results[0]]
