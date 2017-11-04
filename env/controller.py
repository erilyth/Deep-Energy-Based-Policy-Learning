
class Player:
    def __init__(self, x, y, terrain):
        self.x = x
        self.y = y
        self.terrain = terrain

    def action(self, x_m, y_m):
        self.x += x_m
        self.y += y_m
        self.x = min(self.terrain.bounds_x[1], max(self.x, self.terrain.bounds_x[0]))
        self.y = min(self.terrain.bounds_y[1], max(self.y, self.terrain.bounds_y[0]))
        return self.terrain.getreward()
