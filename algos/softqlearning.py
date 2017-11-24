import random
import numpy as np
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from qnet import QNet
from policynet import PolicyNet

from env.terrain import Terrain

from torch import optim

class SoftQLearning:
    def __init__(self):
        self.dO = 2 # Observation space dimensions
        self.dA = 2 # Action space dimensions

        self.criterion = nn.MSELoss()

        self.q_net = QNet()
        self.q_optimizer = optim.SGD(self.q_net.parameters(), lr=0.0001)
        self.policy_net = PolicyNet()
        self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=0.0001)
        self.terrain = Terrain()

        self.replay_buffer_maxlen = 50
        self.replay_buffer = []
        self.exploration_prob = 0.4
        self.alpha = 1.0

        self.action_set = []
        for j in range(32):
            self.action_set.append((math.sin((3.14*2/32)*j), math.cos((3.14*2/32)*j)))

    def forward_QNet(self, obs, action):
        inputs = Variable(torch.FloatTensor([obs + action]))
        q_pred = self.q_net(inputs)
        return q_pred

    def forward_PolicyNet(self, obs):
        inputs = Variable(torch.FloatTensor([obs]))
        action_pred = self.policy_net(inputs)
        return action_pred

    def collect_samples(self):
        self.replay_buffer = []
        self.terrain.resetgame()
        while(1):
            self.terrain.plotgame()
            current_state = self.terrain.player.getposition()
            # TODO: Use action from network here
            best_action = self.action_set[0]
            for j in range(32):
                # Sample 32 actions and use them in the next state to get maximum Q_value
                action_temp = self.action_set[j]
                print self.forward_QNet(current_state, action_temp).data.numpy()[0][0]
                if self.forward_QNet(current_state, action_temp).data.numpy()[0][0] > self.forward_QNet(current_state, best_action).data.numpy()[0][0]:
                    best_action = action_temp
            print "Exploration prob:", self.exploration_prob
            if random.uniform(0.0, 1.0) < self.exploration_prob:
                x_val = random.uniform(-1.0, 1.0)
                best_action = (x_val, random.choice([-1.0, 1.0])*math.sqrt(1.0 - x_val*x_val))
            print "Action:", best_action
            current_reward = self.terrain.player.action(best_action)
            print "Reward:", current_reward
            next_state = self.terrain.player.getposition()
            self.replay_buffer.append([current_state, best_action, current_reward, next_state])
            if self.terrain.checkepisodeend() or len(self.replay_buffer) > self.replay_buffer_maxlen:
                self.terrain.resetgame()
                break

    def train_network(self):
        for t in range(50):
            i = random.randint(0, len(self.replay_buffer)-1)
            current_state = self.replay_buffer[i][0]
            current_action = self.replay_buffer[i][1]
            current_reward = self.replay_buffer[i][2]
            next_state = self.replay_buffer[i][3]
            best_q_val_next = -1000
            for j in range(32):
                # Sample 16 actions and use them in the next state to get maximum Q_value
                # TODO: Needs to be changed
                action_temp = self.action_set[j]
                best_q_val_next = max(best_q_val_next, self.forward_QNet(next_state, action_temp).data.numpy()[0][0])
            predicted_q = self.forward_QNet(current_state, current_action)
            expected_q = current_reward + 0.99 * best_q_val_next
            #print "Expected Q:", expected_q
            expected_q = (1-self.alpha) * predicted_q.data.numpy()[0][0] + self.alpha * expected_q
            expected_q = Variable(torch.FloatTensor([[expected_q]]))
            self.policy_optimizer.zero_grad()
            self.q_optimizer.zero_grad()
            loss = self.criterion(predicted_q, expected_q)
            loss.backward()
            self.q_optimizer.step()

softqlearning = SoftQLearning()
while(1):
    if softqlearning.exploration_prob > 0.1:
        softqlearning.exploration_prob *= 0.9
    softqlearning.collect_samples()
    softqlearning.train_network()
