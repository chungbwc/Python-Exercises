import cv2
import numpy as np
import pygame
import pygame.camera
import pygame.font

from facial_fer_model import FacialExpressionRecog

FACTOR = 2
SMALL = (640, 360)
CAP_SIZE = (SMALL[0] * FACTOR, SMALL[1] * FACTOR)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

pygame.init()
pygame.camera.init()
pygame.font.init()

font = pygame.font.SysFont("Arial", 24)
screen = pygame.display.set_mode(CAP_SIZE, pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("Facial Expression")

cameras = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cameras[0], CAP_SIZE)
cam.start()
cam.set_controls(hflip=True, vflip=False)

model_file = "data/face_detection_yunet_2023mar.onnx"
fer_file = "data/facial_expression_recognition_mobilefacenet_2022july.onnx"

model = cv2.FaceDetectorYN.create(model=model_file,
                                  config="",
                                  input_size=(320, 320),
                                  score_threshold=0.5,
                                  backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                                  target_id=cv2.dnn.DNN_TARGET_CPU)

model.setInputSize(SMALL)

fer_model = FacialExpressionRecog(modelPath=fer_file,
                                  backendId=cv2.dnn.DNN_BACKEND_DEFAULT,
                                  targetId=cv2.dnn.DNN_TARGET_CPU)

running = True
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

while running:
    surface = cam.get_image()
    temp = pygame.surfarray.pixels3d(surface).copy()
    frame = np.array(temp)
    frame = cv2.transpose(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, SMALL)

    ok, faces = model.detect(frame)

    fer_type = ""

    if faces is not None:
        det = faces[0]
        det = det[:-1]

        fer_type = fer_model.infer(frame, det)

        corners = np.array(det[:4]).astype(np.int32).reshape((2, 2))
        corners = corners * FACTOR
        rect = pygame.Rect(corners[0], corners[1])
        pygame.draw.rect(surface,
                         BLUE,
                         rect,
                         2)

        text = font.render(fer_type,
                           True,
                           BLUE)

        surface.blit(text, corners[0])

    screen.blit(surface, (0, 0))
    pygame.display.update()
    clock.tick()

    events = pygame.event.get()
    for e in events:
        if (e.type == pygame.QUIT or
                (e.type == pygame.KEYDOWN and
                 e.key == pygame.K_ESCAPE)):
            running = False

cam.stop()
pygame.mouse.set_visible(True)
pygame.quit()
