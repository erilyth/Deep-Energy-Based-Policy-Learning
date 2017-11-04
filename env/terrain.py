import matplotlib.pyplot as plt
import random

plt.ion()

from controller import Player

class Terrain:
    def __init__(self):
        self.reward_locs = [[-10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [10.0, 10.0]]
        self.reward_range = 1.0
        self.reward_other = -0.5
        self.reward_goal = 30.0
        self.bounds_x = [-12.0, 12.0]
        self.bounds_y = [-12.0, 12.0]
        self.player = Player(0.0, 0.0, self)

    def getreward(self):
        for x_pos, y_pos in self.reward_locs:
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                return self.reward_goal
        return self.reward_other

    def plotgame(self):
        plt.clf()
        for x_pos, y_pos in self.reward_locs:
            plt.plot([x_pos,], [y_pos,], marker='o', markersize=3, color="green")
        print self.player.x, self.player.y
        plt.plot([self.player.x,], [self.player.y,], marker='x', markersize=3, color="red")
        plt.pause(0.005)

"""
terrain = Terrain()
while(1):
    terrain.plotgame()
    print terrain.player.action(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
"""
