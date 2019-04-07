import pygame
class Gambling:
    def __init__(self):
        pass
    def fight(self):
        return [pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_SPACE, "mod":0, "unicode":u' '})]
    def say(self):
        print("halo")