import cv2
import mediapipe as mp
import datetime
import queue

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL = "data/blaze_face_short_range.tflite"


class MPFaceBox:
    def __init__(self, sc):
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.detection)
        self.start = datetime.datetime.now().timestamp()
        self.detector = FaceDetector.create_from_options(options)
        self.scale = sc
        self.qFace = queue.Queue(maxsize=5)
        return

    def detection(self,
                  result: FaceDetectorResult,
                  output_image: mp.Image,
                  timestamp_ms: int):
        res = result.detections
        boxes = []
        for f in res:
            box = f.bounding_box
            boxes.append([box.origin_x * self.scale,
                          box.origin_y * self.scale,
                          box.width * self.scale,
                          box.height * self.scale])

        self.qFace.put(boxes)
        return

    def detect(self, img):
        size = (round(img.shape[1] / self.scale),
                round(img.shape[0] / self.scale))
        small = cv2.resize(img, size)
        time = datetime.datetime.now().timestamp()
        ts = round((time - self.start) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=small)
        self.detector.detect_async(mp_image, ts)
        return

    def getBoxes(self):
        result = self.qFace.get()
        self.qFace.task_done()
        return result
