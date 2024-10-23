import cv2
import numpy as np
import pygame
import pygame.camera
import pygame.font

from ultralytics import YOLO

CAP_SIZE = (1280, 720)
FPS = 60
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

model = YOLO("yolo11n-seg.pt")
labels = model.model.names

pygame.init()
pygame.camera.init()
pygame.font.init()

font = pygame.font.SysFont("Arial", 18)
cameras = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cameras[0], CAP_SIZE)
cam.start()
cam.set_controls(hflip=True, vflip=False)

screen = pygame.display.set_mode(CAP_SIZE, pygame.HWSURFACE | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
pygame.display.set_caption("YOLO")

running = True

while running:
    surface = cam.get_image()
    temp = pygame.surfarray.pixels3d(surface).copy()
    temp = np.array(temp)
    temp = cv2.transpose(temp)
    frame = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

    results = model.predict(frame, device="mps")

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        for mask, cls in zip(masks, clss):
            label = labels[round(cls)]
            points = mask.round().tolist()

            arc = 0.01 * cv2.arcLength(mask, True)
            approx = cv2.approxPolyDP(mask, arc, True)
            #            approx = cv2.convexHull(approx)
            approx = np.squeeze(approx).astype(np.int32).tolist()

            pygame.draw.polygon(surface,
                                RED,
                                approx,
                                0)

            text = font.render(label,
                               True,
                               BLACK)

            surface.blit(text, points[0])

    screen.blit(surface, (0, 0))
    pygame.display.update()
    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif (e.type == pygame.KEYDOWN and
              e.key == pygame.K_ESCAPE):
            running = False

cam.stop()
pygame.quit()
