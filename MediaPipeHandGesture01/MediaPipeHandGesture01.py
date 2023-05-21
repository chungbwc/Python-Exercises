import mediapipe as mp
import os
import cv2
import math
import numpy as np
import pygame
from datetime import datetime
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

dimen = (1280, 720)
fps = 30

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

hand_landmarks_proto = None
category = "None"
score = 0


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #    print('gesture recognition result: {}'.format(result))
    global hand_landmarks_proto, category, score
    hand_landmarks_proto = None
    category = "None"
    gestures = result.gestures

    if len(gestures) > 0:
        category = gestures[0][0].category_name
        score = gestures[0][0].score
        #        if category != "None":
        #            print(gestures[0][0].score, gestures[0][0].category_name)
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in result.hand_landmarks[0]
        ])
    return


pygame.init()
pygame.font.init()
font = pygame.font.SysFont("calibri", 32)
screen = pygame.display.set_mode(dimen,
                                 pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("hand gesture")
running = True
clock = pygame.time.Clock()

model_path = os.getcwd() + os.path.sep + "gesture_recognizer.task"
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)
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
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    now = math.floor(datetime.now().timestamp() * 1000)
    recognizer.recognize_async(mp_image, now - start)

    if hand_landmarks_proto is not None:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    drawing = cv2.transpose(frame)
    view = pygame.pixelcopy.make_surface(drawing)
    screen.blit(view, [0, 0])
    text = font.render(category, True, (200, 0, 0))
    screen.blit(text, [40, 50])
    if category != "None":
        text = font.render("{:.3f}".format(score), True, (255, 255, 0))
        screen.blit(text, [250, 50])

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
