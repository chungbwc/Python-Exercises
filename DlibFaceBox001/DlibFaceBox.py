import cv2
import dlib


class DlibFaceBox:
    def __init__(self, sc):
        self.detector = dlib.get_frontal_face_detector()
        self.scale = sc
        return

    def detect(self, img):
        size = (round(img.shape[1] / self.scale),
                round(img.shape[0] / self.scale))
        small = cv2.resize(img, size)
        dets = self.detector(small, 0)
        boxes = []

        for i, d in enumerate(dets):
            x1 = d.left() * self.scale
            y1 = d.top() * self.scale
            x2 = (d.right() - d.left()) * self.scale
            y2 = (d.bottom() - d.top()) * self.scale
            boxes.append([x1, y1, x2, y2])

        return boxes
