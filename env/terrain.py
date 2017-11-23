import matplotlib.pyplot as plt
import random
import numpy as np

plt.ion()

from controller import Player

class Terrain:
    def __init__(self):
        self.reward_locs = [[-10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [10.0, 10.0]]
        self.reward_range = 1.0
        self.reward_goal = 30.0
        self.bounds_x = [-12.0, 12.0]
        self.bounds_y = [-12.0, 12.0]
        self.player = Player(0.0, 0.0, self)

    def getreward(self):
        reward = 0.0
        for x_pos, y_pos in self.reward_locs:
            reward -= np.sqrt((self.player.x - x_pos) ** 2 + (self.player.y - y_pos) ** 2) / 50
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                reward += self.reward_goal
        return reward

    def checkepisodeend(self):
        for x_pos, y_pos in self.reward_locs:
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                return 1
        return 0

    def plotgame(self):
        plt.clf()
        for x_pos, y_pos in self.reward_locs:
            plt.plot([x_pos,], [y_pos,], marker='o', markersize=3, color="green")
        plt.plot([self.player.x,], [self.player.y,], marker='x', markersize=3, color="red")
        plt.pause(0.001)

    def resetgame(self):
        self.player = Player(0.0, 0.0, self)
        plt.close()

"""
terrain = Terrain()
while(1):
    terrain.plotgame()
    print terrain.player.action(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
"""
