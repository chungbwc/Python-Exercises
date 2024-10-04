import cv2
import numpy as np

MODEL = "data/face_detection_yunet_2023mar_int8.onnx"
# MODEL = "data/face_detection_yunet_2023mar.onnx"


class YuNetFaceBox:
    def __init__(self, sc):
        self.detector = cv2.FaceDetectorYN.create(model=MODEL,
                                                  config="",
                                                  input_size=(320, 320),
                                                  score_threshold=0.5,
                                                  backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                                                  target_id=cv2.dnn.DNN_TARGET_CPU)
        self.scale = sc
        return

    def detect(self, img):
        size = (round(img.shape[1] / self.scale),
                round(img.shape[0] / self.scale))
        small = cv2.resize(img, size)
        self.detector.setInputSize(size)
        ok, faces = self.detector.detect(small)
        boxes = []
        if faces is not None:
            for idx, det in enumerate(faces):
                corners = np.array(det[:4]).astype(np.int32).reshape((2, 2))
                corners = corners * self.scale
                boxes.append([corners[0][0],
                              corners[0][1],
                              corners[1][0],
                              corners[1][1]])

        return boxes
