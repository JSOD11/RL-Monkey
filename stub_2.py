# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

import matplotlib.pyplot as plt

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
#from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0

        self.epsilon = 0.001
        self.alpha = 0.2
        self.gamma = 0.8
        self.v = 0

        # gravity data gathered from trials
        self.v_list = list(range(40))

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE, len(self.v_list)))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.v = 0

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y, self.v_list.index(self.v))

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.

        d_state = self.discretize_state(state)

        if not self.last_state:
            self.last_state = d_state
            self.last_action = 0
            self.v = state["monkey"]["vel"]+20

            return self.last_action

        q_values = [self.Q[0][d_state], self.Q[1][d_state]]
        action = np.argmax(q_values)
        action_val = np.max(q_values)

        self.Q[self.last_action][self.last_state] += self.alpha*(self.last_reward + self.gamma*action_val - self.Q[self.last_action][self.last_state])

        if npr.rand() < self.epsilon:
            self.last_action = int(npr.rand() < 0.5)
        else:
            self.last_action = action

        self.last_state = d_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    hist_of_hists = []

    epochs = 1

    for run in range(epochs):
        # Empty list to save history.
        hist = []

        # Run games. You can update t_len to be smaller to run it faster.
        run_games(agent, hist, 100, 100)
        #print(hist)
        print()
        print("Mean score for this monkey: " + str(np.mean(hist)))
        print("Score variance for this monkey: " + str(np.var(hist)))
        print()

        for score in hist:
            hist_of_hists.append(score)

        # Save history. 
        np.save('hist', np.array(hist))
        plt.plot(hist)
        plt.title("Monkey Version 2 - Jump Delay and Velocity Information")
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.savefig("monkey2.png")
        plt.show()
    
    print("Mean score over " + str(epochs) + " epochs for this monkey: " + str(np.mean(hist)))
    print("Score variance over " + str(epochs) + " epochs for this monkey: " + str(np.var(hist)))