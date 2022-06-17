import torch
import random
import numpy as np
from collections import deque
from game import CopsRobbersAI, Direction, Point, BLOCK_SIZE
from modelt import Linear_QNett, QTrainert
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 #
        self.memory = deque(maxlen = MAX_MEMORY)
        self.memoryt = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 4)
        self.modelt = Linear_QNett(16, 256, 4)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        self.trainert = QTrainert(self.modelt, lr = LR, gamma = self.gamma)
    
    def get_state(self, game):
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # thief location
            game.thief.x < game.police.x, # thief left
            game.thief.x > game.police.x, # thief right
            game.thief.y < game.police.y, # thief up
            game.thief.y > game.police.y,  # thief down

            game.door[0].x < game.police.x, # door left
            game.door[0].x > game.police.x, # door right
            game.door[0].y < game.police.y, # door up
            game.door[0].y > game.police.y,  # door down

            game.door[1].x < game.police.x, # door left
            game.door[1].x > game.police.x, # door right
            game.door[1].y < game.police.y, # door up
            game.door[1].y > game.police.y,  # door down
        ]
        return np.array(state, dtype = int)

    def get_statet(self, game):
        dir_l = game.directiont == Direction.LEFT
        dir_r = game.directiont == Direction.RIGHT
        dir_u = game.directiont == Direction.UP
        dir_d = game.directiont == Direction.DOWN

        state = [
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # thief location
            game.police.x < game.thief.x, # police left
            game.police.x > game.thief.x, # police right
            game.police.y < game.thief.y, # police up
            game.police.y > game.thief.y,  # police down

            game.door[0].x < game.thief.x, # door left
            game.door[0].x > game.thief.x, # door right
            game.door[0].y < game.thief.y, # door up
            game.door[0].y > game.thief.y,  # door down

            game.door[1].x < game.thief.x, # door left
            game.door[1].x > game.thief.x, # door right
            game.door[1].y < game.thief.y, # door up
            game.door[1].y > game.thief.y,  # door down
        ]
        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    ############
    def remembert(self, state, action, reward, next_state, done):
        self.memoryt.append((state, action, reward, next_state, done))
###################

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    ################
    def train_long_memoryt(self):
        if len(self.memoryt) > BATCH_SIZE:
            mini_sample = random.sample(self.memoryt, BATCH_SIZE)
        else:
            mini_sample = self.memoryt
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainert.train_step(states, actions, rewards, next_states, dones)
    ###################
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    ###############
    def train_short_memoryt(self, state, action, reward, next_state, done):
        self.trainert.train_step(state, action, reward, next_state, done)
    #################

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_actiont(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.modelt(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    recordt = 0

    agent = Agent()
    game = CopsRobbersAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        state_oldt = agent.get_statet(game)
        final_movet = agent.get_actiont(state_oldt)

        reward, done, score, scoret, rewardt = game.play_step(final_move, final_movet)
        state_new = agent.get_state(game)
        state_newt = agent.get_statet(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.train_short_memoryt(state_oldt, final_movet, rewardt, state_newt, done)
        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        agent.remembert(state_oldt, final_movet, rewardt, state_newt, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.train_long_memoryt()

            if score > record:
                record = score
                agent.model.save()
            if scoret > recordt:
                recordt = scoret
                agent.modelt.save()
            print('game', agent.n_games, 'score', score, 'record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()