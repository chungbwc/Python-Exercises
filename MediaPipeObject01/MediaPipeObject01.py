import mediapipe as mp
import os
import cv2
import math
import pygame
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

dimen = (1280, 720)
fps = 30
scaling = 2
small = (dimen[0] // 2, dimen[1] // 2)

BaseOptions = mp.tasks.BaseOptions

DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

category = "None"
bbox = None


def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global category, bbox
    obj = result.detections[0]
    bbox = obj.bounding_box
    category = obj.categories[0].category_name
    #    print('detection result: {}'.format(result))
    return


pygame.init()
pygame.font.init()
font = pygame.font.SysFont("calibri", 32)
screen = pygame.display.set_mode(dimen,
                                 pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("hand gesture")
running = True
clock = pygame.time.Clock()

model_path = os.getcwd() + os.path.sep + "efficientdet_lite0.tflite"
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=1,
    result_callback=print_result)

detector = ObjectDetector.create_from_options(options)
start = math.floor(datetime.now().timestamp() * 1000)

cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimen[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimen[1])
cap.set(cv2.CAP_PROP_FPS, fps)

while running:
    ok, frame = cap.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    small_frame = cv2.resize(frame, small)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
    now = math.floor(datetime.now().timestamp() * 1000)
#    bbox = None
    detector.detect_async(mp_image, now - start)

    drawing = cv2.transpose(frame)
    view = pygame.pixelcopy.make_surface(drawing)
    screen.blit(view, [0, 0])

    text = font.render(category, True, (200, 0, 0))
    screen.blit(text, [40, 50])

    if bbox is not None:
        pygame.draw.rect(screen, (255, 255, 0),
                         pygame.Rect(bbox.origin_x * scaling,
                                     bbox.origin_y * scaling,
                                     bbox.width * scaling, bbox.height * scaling), 1)

    pygame.display.update()
    clock.tick(fps)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

pygame.font.quit()
pygame.quit()
cap.release()
