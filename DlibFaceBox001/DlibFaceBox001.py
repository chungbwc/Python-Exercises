import pygame
import pygame.font
import queue

from Capture import Capture
from DlibFaceBox import DlibFaceBox

CAP_SIZE = (1280, 720)

FPS = 60
FACTOR = 2


def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(CAP_SIZE, pygame.DOUBLEBUF | pygame.HWSURFACE)
    pygame.display.set_caption("Threading")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    running = True
    pygame.mouse.set_visible(False)

    qSf = queue.Queue(maxsize=5)
    qCV = queue.Queue(maxsize=5)

    cap = Capture(CAP_SIZE, qSf, qCV)
    face = DlibFaceBox(FACTOR)

    while running:
        fps = round(clock.get_fps())
        text = font.render(str(fps), True, (255, 0, 0))

        surf = qSf.get()
        screen.blit(surf, (0, 0))
        qSf.task_done()

        screen.blit(text, (20, 20))

        img = qCV.get()
        qCV.task_done()
        faces = face.detect(img)

        if faces is not None:
            for f in faces:
                rect = pygame.Rect(f[0], f[1], f[2], f[3])
                pygame.draw.rect(screen,
                                 (0, 0, 255),
                                 rect,
                                 2)

        pygame.display.update()
        clock.tick(FPS)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif (e.type == pygame.KEYDOWN and
                  e.key == pygame.K_ESCAPE):
                running = False

    cap.stop()
    pygame.mouse.set_visible(True)
    pygame.quit()
    return


if __name__ == "__main__":
    main()
