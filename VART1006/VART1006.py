#!/usr/bin/env python3

import cv2
import mediapipe as mp
import pygame
import pygame.font
import numpy as np
import os

MAX_FILE = 29

myDir = os.getcwd()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_spec = mp.solutions.drawing_utils.DrawingSpec
mp_pose = mp.solutions.pose

mp_landmark = mp_drawing_spec(
    color=(10, 10, 10),
    thickness=1,
    circle_radius=0
)

mp_connection = mp_drawing_spec(
    color=(0, 0, 255),
    thickness=2,
    circle_radius=0
)

dimen1 = (1280, 720)
dimen2 = (1920, 1080)
fps = 30
pygame.init()
pygame.font.init()

screen = pygame.display.set_mode(dimen1, pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("dance pose")
running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("futura", 32)

# cap = cv2.VideoCapture(cv2.CAP_ANY)
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
    print("Camera error")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimen1[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimen1[1])
cap.set(cv2.CAP_PROP_FPS, fps)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

back = np.zeros((dimen1[1], dimen1[0], 3), dtype=np.uint8)
cnt = 0

while running:
    screen.fill((0, 0, 0))
    success, frame = cap.read()

    if frame is None:
        print("Frame not ready")
        continue

    back[:] = (255, 255, 255)
    flip = cv2.flip(frame, 1)

    image = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    mp_drawing.draw_landmarks(
        back,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_landmark,
        connection_drawing_spec=mp_connection
    )

    mp_drawing.draw_landmarks(
        flip,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_landmark,
        connection_drawing_spec=mp_connection
    )

    drawing = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
    drawing = cv2.transpose(drawing)
    view1 = pygame.pixelcopy.make_surface(drawing)

    image = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    image = cv2.transpose(image)
    view2 = pygame.pixelcopy.make_surface(image)

    count = font.render(str(cnt + 1), True, (255, 0, 0))
    screen.blit(view2, [0, 0])
    screen.blit(count, [50, 50])
    pygame.display.update()
    clock.tick(fps)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            cnt = cnt + 1
            fileName = os.path.join(myDir, "data", "mm" + f'{cnt:03d}' + ".png")
#            fileName = "data/mm" + f'{cnt:03d}' + ".png"
            print(fileName)
            big = pygame.transform.smoothscale(view1, dimen2)
            pygame.image.save(big, fileName)
            if cnt > MAX_FILE:
                running = False

pygame.quit()
cap.release()
