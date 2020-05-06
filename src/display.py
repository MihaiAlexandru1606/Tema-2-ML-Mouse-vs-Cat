import pygame
from pygame import transform, image
import time

from src.map import Map

WIDTH_CELL = 25
HEIGHT_CELL = 25
ICON_PHOTO = "./image/icon.jpg"
MOUSE_PHOTO = "./image/mouse.png"
CHEESE_PHOTO = "./image/cheese.png"
CAT_PHOTO = "./image/cat.png"
WALL_PHOTO = "./image/wall-2.jpg"
BG_COLOR = [70, 191, 238, 255]


class Display(object):
    IMAGE_CAT = transform.scale(image.load(CAT_PHOTO), (WIDTH_CELL, HEIGHT_CELL))
    IMAGE_MOUSE = transform.scale(image.load(MOUSE_PHOTO), (WIDTH_CELL, HEIGHT_CELL))
    IMAGE_CHEESE = transform.scale(image.load(CHEESE_PHOTO), (WIDTH_CELL, HEIGHT_CELL))
    IMAGE_WALL = transform.scale(image.load(WALL_PHOTO), (WIDTH_CELL, HEIGHT_CELL))

    def __init__(self, game_map: Map, time_wait=0.5):
        self.game_map = game_map
        pygame.init()
        pygame.display.set_icon(image.load(ICON_PHOTO))
        pygame.display.set_caption("Mouse Vs Cat")

        size = (game_map.width * WIDTH_CELL, game_map.height * HEIGHT_CELL)

        self.screen = pygame.display.set_mode(size)
        self.time_wait = time_wait

    def render(self):
        self.screen.fill(BG_COLOR)

        self.screen.blit(Display.IMAGE_CAT,
                         (self.game_map.position_cat[1] * WIDTH_CELL, self.game_map.position_cat[0] * HEIGHT_CELL))
        self.screen.blit(Display.IMAGE_MOUSE,
                         (self.game_map.position_mouse[1] * WIDTH_CELL, self.game_map.position_mouse[0] * HEIGHT_CELL))

        for position_cheese in self.game_map.position_cheeses:
            self.screen.blit(Display.IMAGE_CHEESE,
                             (position_cheese[1] * WIDTH_CELL, position_cheese[0] * HEIGHT_CELL))

        for position_obstacle in self.game_map.position_obstacles:
            self.screen.blit(Display.IMAGE_WALL,
                             (position_obstacle[1] * WIDTH_CELL, position_obstacle[0] * HEIGHT_CELL))

        pygame.display.update()
        time.sleep(self.time_wait)

    @staticmethod
    def is_close() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

    @staticmethod
    def close_display():
        pygame.display.quit()
