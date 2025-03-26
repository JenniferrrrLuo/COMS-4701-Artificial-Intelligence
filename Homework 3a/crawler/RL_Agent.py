"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A Q-learning agent for a stochastic task environment
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def choose_action(self, state, valid_actions):
        """ Choose an action using epsilon-greedy selection.

        Args:
            state (tuple): Current robot state.
            valid_actions (list): A list of possible actions.
        Returns:
            action (string): Action chosen from valid_actions.
        
        First write the choose action() method, which performs ε-greedy action selection given the state
        and valid actions list. It should make reference to Qvalues if deciding to behave greedily.
        """
        # TODO
        
        if random.random() > self.epsilon: 
            best_action = None
            max_value = 0
            for action in valid_actions:
                Qvalue = self.Qvalues.get((state, action),0)
                if Qvalue > max_value:
                    max_value = Qvalue
                    best_action = action 
        else: 
            best_action = random.choice(valid_actions)
        
        return best_action
    

    def update(self, state, action, reward, successor, valid_actions):
        """ Update self.Qvalues for (state, action) given reward and successor.

        Args:
            state (tuple): Current robot state.
            action (string): Action taken at state.
            reward (float): Reward given for transition.
            successor (tuple): Successor state.
            valid_actions (list): A list of possible actions at successor state.
            
        Next, write the update() method, which makes a Q-learning update to the appropriate Q-value given
        all components of a single transition and the valid actions of the successor state. As in Part 1
        above, if the successor is None, you may set its “Q-value” to 0.
        """
        # TODO
        
        max_next_value = float('-inf')
        
        for next_action in valid_actions:
            next_Qvalue = self.Qvalues.get((successor, next_action),0)
            if next_Qvalue > max_next_value:
                max_next_value = next_Qvalue
        
        current_Qvalue = self.Qvalues.get((state,action),0)
        
        self.Qvalues[(state, action)] = current_Qvalue + self.alpha * (reward + self.gamma * max_next_value - current_Qvalue)
        
        
        
        