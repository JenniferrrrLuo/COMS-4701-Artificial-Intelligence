"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None

    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass

    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass

    def value_iteration(self, valid_actions, transition):
        """ Computes all optimal values using value iteration and stores them in self.values.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.

        First write the value iteration() method. This should run value iteration to find the optimal
        values V∗ for all states and store them in the values dictionary. Two Callables (function handles)
        are given as arguments: valid actions returns a list of actions given a state, and transition
        returns the successor state and reward given a state and action (all transitions are deterministic).
        If None is returned as a successor state, you can use 0 as its “value”. Convergence may occur when
        the maximum change in any state value is no greater than 10−6
        """
        # TODO
        
        threshold = 10**(-6)
        
        while True: 
            delta = 0
            for state in self.states: 
                action_values = []
                for action in valid_actions(state):
                     value = 0
                     nextState, reward = transition(state, action)
                     value += reward + self.gamma * self.values.get(nextState, 0)
                     action_values.append(value)
                final_value = max(action_values)
                delta = max(delta, abs(self.values[state] - final_value))
                self.values[state] = final_value
            if delta < threshold:
                break
        
    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        
        After we run value iteration, we need to derive a policy π∗
        . Write policy extraction(), which will store the optimal actions for all states in the policy dictionary. Once you finish this, you can
        test your implementation by running python crawler.py. If all goes well, your robot should be
        able to start moving across the screen with its newfound policy.
        
        """
        # TODO      
        for state in self.states: 
            best_action = None
            max_value = float('-inf')
            for action in valid_actions(state):
                nextState, reward = transition(state, action)
                value = reward + self.gamma * self.values.get(nextState,0)
                if value > max_value:
                    max_value = value
                    best_action = action 
            self.policy[state] = best_action
    
    
    
    
