from ultralytics import YOLO
import cv2
import pygame
import pygame.font
import pygame.camera
import numpy as np

CAP_SIZE = (1280, 720)
FPS = 60

model = YOLO("yolo11n.pt")

pygame.init()
pygame.camera.init()
pygame.font.init()

font = pygame.font.SysFont("Arial", 16)
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

    results = model.track(source=frame,
                          persist=True,
                          #                      tracker="bytetrack.yaml",
                          device="mps")

    if results[0].boxes.is_track:
        ids = results[0].boxes.id.cpu().tolist()
        boxes = results[0].boxes.xywh.cpu().tolist()
        cls = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.cpu().conf.tolist()

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            w = round(w)
            h = round(h)
            x = round(x - w / 2.0)
            y = round(y - h / 2.0)

            id = round(ids[i])
            name = results[0].names[round(cls[i])]

            txt = str(id) + " " + name + " (" + "{:.2f}".format(round(confs[i], 2)) + ")"
            text = font.render(txt, True, (0, 0, 255))

            pygame.draw.rect(surface,
                             (255, 0, 0),
                             pygame.Rect(x, y, w, h),
                             2)

            surface.blit(text, (x, y))

    # output = results[0].plot(line_width=2)

    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # output = cv2.transpose(output)
    # surf = pygame.pixelcopy.make_surface(output)

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
