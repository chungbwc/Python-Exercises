import pygame
import pygame.camera
import cv2
import numpy as np
import asyncio


class Capture:
    def __init__(self, size, queue):
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        self.cam = pygame.camera.Camera(cameras[0], size)
        self.cam.start()
        self.cam.set_controls(hflip=True, vflip=False)
        self.surface = pygame.Surface(size)
        self.cv = np.zeros((size[1], size[0], 3))
        self.running = True
        self.queue = queue
        print("Capture inited...")

        return

    async def run(self):
        while self.running:
            self.surface = self.cam.get_image().copy()
            await self.queue.put(self.surface)
            temp = pygame.surfarray.pixels3d(self.surface).copy()
            temp = np.array(temp)
            temp = cv2.transpose(temp)
            self.cv = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

        self.cam.stop()
        return

    def stop(self):
        self.running = False
        return
