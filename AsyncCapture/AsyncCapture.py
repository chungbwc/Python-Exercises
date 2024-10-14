import pygame
import pygame.font
from Capture import Capture
import asyncio

CAP_SIZE = (1280, 720)
FPS = 60


async def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(CAP_SIZE, pygame.DOUBLEBUF)
    pygame.display.set_caption("Threading")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    running = True
    pygame.mouse.set_visible(False)

    queue = asyncio.Queue(maxsize=1)
    cap = Capture(CAP_SIZE, queue)
    task = asyncio.create_task(cap.run())
    print("task created...")
    await asyncio.sleep(0.2)

    while running:
        fps = round(clock.get_fps())
        text = font.render(str(fps), True, (255, 0, 0))
        surf = await queue.get()
        screen.blit(surf, (0, 0))
        queue.task_done()

        screen.blit(text, (20, 20))
        pygame.display.update()
        clock.tick(FPS)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif (e.type == pygame.KEYDOWN and
                  e.key == pygame.K_ESCAPE):
                running = False

    await queue.join()
    task.cancel()

    cap.stop()
    pygame.mouse.set_visible(True)
    pygame.quit()
    return


if __name__ == "__main__":
    asyncio.run(main())
