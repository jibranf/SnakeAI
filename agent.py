import torch
import random
import numpy as np
from game import BLOCK_SIZE, SnakeGameAI, Direction, Point
from collections import deque # data structure to support memory
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # parameter to control randomness
        self.gamma = 0.9 # discount rate in Belman Equation, must be smaller than 1 typically 0.8/0.9
        self.memory = deque(maxlen = MAX_MEMORY) # automatically remove elements from left side of structure when its full
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma = self.gamma)


    def get_state(self, game):
        head = game.snake[0]

        # checking for danger at different points
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        # booleans for current direction of snake
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_down and game.is_collision(point_down)) or
            (dir_up and game.is_collision(point_up)),
        
            # danger right
            (dir_right and game.is_collision(point_down)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_right)),


            # danger left
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)) or
            (dir_down and game.is_collision(point_right)) or
            (dir_up and game.is_collision(point_left)),


            # move direction, only one of these is true at one time
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # location of food from head
            game.food.x < game.head.x, # food is on the left
            game.food.y > game.head.x, # food is on the right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y # food is down
        ]

        return np.array(state, dtype = int) # dtype converts booleans into 0s or 1s


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            small_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples

        else:
            small_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*small_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration/exploitation
        self.epsilon = 80 - self.num_games # the more games that are played, the smaller epsilon becomes, and at some point moves will no longer be random
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # picks a random index to change for movement
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0) # calls the forward function in model class
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    # list of scores
    plot_scores =  []
    plot_mean_scores = []
    total_score = 0
    record = 0 # best score
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state of game
        old_state = agent.get_state(game)

        # get move based on this state
        final_move = agent.get_action(old_state)

        #perform move and get new_state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory of agent
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # remember the training and store it in memory
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # train long memory if game is over and plot results
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            # checks for new high score
            if score > record:
                record = score
                agent.model.save()

            print("Game ", agent.num_games, "Score ", score, "Record: ", record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()


