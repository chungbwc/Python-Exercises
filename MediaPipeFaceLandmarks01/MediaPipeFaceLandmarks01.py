import mediapipe as mp
import numpy as np
import os
import cv2
import math
import pygame
from datetime import datetime
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation as R

from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

dimen = (1280, 720)
fps = 30
scaling = 2
small = (dimen[0] // 2, dimen[1] // 2)

BaseOptions = mp.tasks.BaseOptions

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global face_landmarks, face_matrix, face_blend
    face_landmarks = None
    face_matrix = None
    face_blend = None
    if result is not None:
        if len(result.face_landmarks) > 0:
            face_landmarks = result.face_landmarks[0]
            face_matrix = result.facial_transformation_matrixes[0]
            face_blend = result.face_blendshapes[0]
    #    print('face result: {}'.format(result))
    return


pygame.init()
pygame.font.init()
font = pygame.font.SysFont("calibri", 32)
screen = pygame.display.set_mode(dimen,
                                 pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("face")
running = True
clock = pygame.time.Clock()

pooh_file = os.getcwd() + os.path.sep + "Pooh.png"
pooh = pygame.image.load(pooh_file)

model_path = os.getcwd() + os.path.sep + "face_landmarker.task"
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    result_callback=print_result)

landmarker = FaceLandmarker.create_from_options(options)
start = math.floor(datetime.now().timestamp() * 1000)
face_landmarks = None
face_matrix = None
face_blend = None

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
    landmarker.detect_async(mp_image, now - start)

    face_rot = 0
    if face_matrix is not None:
        #        inv_matrix = np.linalg.inv(face_matrix)
        rot = face_matrix[:3, :3]
        r = R.from_matrix(rot)
        r = r.as_euler('zxy', degrees=True)
        face_rot = r[0]

    if face_landmarks is not None:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())

        solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())

    #        solutions.drawing_utils.draw_landmarks(
    #            image=frame,
    #            landmark_list=face_landmarks_proto,
    #            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #            landmark_drawing_spec=None,
    #            connection_drawing_spec=mp.solutions.drawing_styles
    #            .get_default_face_mesh_iris_connections_style())

    drawing = cv2.transpose(frame)
    view = pygame.pixelcopy.make_surface(drawing)
    pooh_view = pygame.transform.rotate(pooh, face_rot)

    w2 = pooh_view.get_rect().width
    h2 = pooh_view.get_rect().height

    screen.blit(view, [0, 0])
    screen.blit(pooh_view, [250 - w2 // 2, 250 - h2 // 2])

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
