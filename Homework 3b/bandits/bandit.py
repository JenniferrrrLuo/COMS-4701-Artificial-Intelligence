from typing import Tuple
from graph import Graph

import numpy as np
import numpy.typing as npt


class Bandit:

    def __init__(
            self,
            graph: Graph,
            conditional_sigma: float,
            strategy: int,
            value: float,
            N: int
    ):
        self.graph = graph
        self.arms = self.graph.arms
        self.edges = self.graph.edges

        self.conditional_sigma = conditional_sigma
        self.strategy = strategy
        self.value = value
        self.N = N

        self.Qvalues = np.zeros(len(self.arms))
        self.arm_counts = np.zeros(len(self.arms))


    def simulate(
        self
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        # TODO
        
        best_arm = self.graph.shortest_path_ind()
    
        regret = np.zeros(self.N)
        
        for t in range(0, self.N):
            if self.strategy == 0:
                arm = self.choose_arm_egreedy()
            elif self.strategy == 1:
                arm = self.choose_arm_edecay(t) 
            elif self.strategy == 2:
                arm = self.choose_arm_ucb(t)
            else:
                raise ValueError
                
            reward = self.pull_arm(arm)
            
            self.Qvalues[arm] = (self.arm_counts[arm] * self.Qvalues[arm] + reward) / (self.arm_counts[arm] + 1)
            
            self.arm_counts[arm] += 1
            
            if arm == best_arm: 
                regret[t] = 0
            else: 
                regret[t] = self.pull_arm(best_arm) - reward
            
        return self.Qvalues, regret


    def choose_arm_egreedy(
        self
    ) -> int:
        # TODO
        epsilon = self.value
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.Qvalues))
        else: 
            return np.argmax(self.Qvalues)

    def choose_arm_edecay(
        self,
        t: int
    ) -> int:
        # TODO
        K = len(self.arms)
        c = self.value
        epsilon = min(1, c * K / (t + 1))
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.Qvalues))
        else: 
            return np.argmax(self.Qvalues)

    def choose_arm_ucb(
        self,
        t: int
    ) -> int:
        # TODO
        c = self.value
        if t == 0: 
            ucb_values = self.Qvalues + c * np.sqrt(np.log(t+1) / (self.arm_counts + 1))
        else: 
            ucb_values = self.Qvalues + c * np.sqrt(np.log(t) / (self.arm_counts + 1))
        return np.argmax(ucb_values)

    def pull_arm(
        self,
        idx: int,
    ) -> float:
        reward = 0
        for i in range(len(self.arms[idx]) - 1):
            mu_edge = self.edges[self.arms[idx][i]][self.arms[idx][i + 1]]["mu"]
            conditional_mean = np.log(mu_edge) - 0.5 * (self.conditional_sigma ** 2)
            reward -= np.exp(conditional_mean + self.conditional_sigma * np.random.randn())
        return reward


    def get_path_mean(
        self,
        idx: int,
    ) -> float:
        return -self.graph.all_path_means[idx]