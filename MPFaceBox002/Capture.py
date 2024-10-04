import pygame
import pygame.camera
import threading
import cv2
import numpy as np


class Capture:
    def __init__(self, size, q1, q2):
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        self.cam = pygame.camera.Camera(cameras[0], size)
        self.cam.start()
        self.cam.set_controls(hflip=True, vflip=False)
        self.surface = pygame.Surface(size)
        self.cv = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.running = True
        self.qSf = q1
        self.qCV = q2
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()
        return

    def run(self):
        while self.running:
            self.surface = self.cam.get_image()
            temp = pygame.surfarray.pixels3d(self.surface).copy()
            temp = np.array(temp)
            temp = cv2.transpose(temp)
            self.cv = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

            self.qSf.put(self.surface.copy())
            self.qCV.put(self.cv.copy())

        self.cam.stop()
        return

    def stop(self):
        with self.lock:
            self.running = False
        return
