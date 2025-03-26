import numpy as np
import numpy.typing as npt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = []):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = ((1 - self.grid) / np.sum(self.grid)).flatten('F')

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [(i, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = []
        for a1, a2 in adjacent:
            if 0 <= a1 < M and 0 <= a2 < N and self.grid[a1, a2] == 0:
                neighbors.append((a1, a2))
        return neighbors

    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return nxn transition matrix, where n = size of grid.
        """
        # TODO
        
        n = self.grid.size
        M, N = self.grid.shape
        
        T = np.zeros((n, n))
        
        for i in range(M):
            for j in range(N):
                free_neighbors = self.neighbors((i, j))
                num_neighbors = len(free_neighbors)
                current_state = i * N + j
                for neighbor in free_neighbors:
                    next_state = neighbor[0] * N + neighbor[1]
                    T[current_state, next_state] = 1 / num_neighbors
                
        return T

    def initO(self):
        """
        Create and return 16xn matrix of observation probabilities, where n = size of grid.
        """
        # TODO
        
        n = self.grid.size
        M, N = self.grid.shape
        O = np.zeros((16, n))
        
        for i in range(M):
            for j in range(N):
                correct_observation = self.find_correct_observation((i, j))
                current_state = i * N + j
                for e in range(16):
                    d = bin(correct_observation ^ e).count('1')
                    O[e, current_state] = (1 - self.epsilon) ** (4 - d) * self.epsilon ** d

        return O

    def find_correct_observation(self, cell):
        i, j = cell
        M, N = self.grid.shape
        # North, East, South, West
        directions = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
        
        observation = 0
        
        for idx, (di, dj) in enumerate(directions):
            if 0 <= di < M and 0 <= dj < N and self.grid[di, dj] == 1:
                observation += 2 ** (3 - idx)
                
        return observation

    """
    4.3 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO
        
        alpha = np.matmul(alpha, self.trans) * self.obs[observation]
        
        return alpha

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """
        # TODO
        
        beta = np.matmul((beta * self.obs[observation]).T, self.trans.T)
        
        return beta

    def filtering(self, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Alpha vectors at each timestep.
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO
        
        T = len(observations)
        N = self.grid.size
        
        alphas = np.zeros((T, N))
        alphas[0, :] = self.init
        
        for t in range(1, T):
            alphas[t, :] = self.forward(alphas[t-1, :], observations[t])
        
        return alphas, alphas / alphas.sum(axis=1, keepdims=True)


    def smoothing(self, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Beta vectors at each timestep.
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO
        
        T = len(observations)
        N = self.grid.size
        alphas, filtered_states = self.filtering(observations)
        
        betas = np.zeros((T, N))
        betas[-1, :] = np.ones(N)
        
        for t in range(T-2, -1, -1):
            betas[t, :] = self.backward(betas[t+1, :], observations[t+1])
    
        smoothed_states = alphas * betas
        for i in range(T):
            row_sum = smoothed_states[i, :].sum()
            if row_sum > 0:
                smoothed_states[i, :] /= row_sum
        
        return betas, smoothed_states

    """
    4.4. Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        """Learn observation probabilities using the Baum-Welch algorithm.
        Updates self.obs in place.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Learned 16xn matrix of observation probabilities, where n = size of grid.
          list[float]: List of data likelihoods at each iteration.
        """
        # TODO
        
        log_likelihoods = []
        convergence_threshold = 1e-3
        has_converged = False
        
        while not has_converged:

            betas, smoothed_states = self.smoothing(observations)
            
            new_obs_probs = np.zeros_like(self.obs)
            for obs in range(16):
                for t, observation in enumerate(observations):
                    if observation == obs:
                        new_obs_probs[obs, :] += smoothed_states[t, :]
            
            new_obs_probs_sum = new_obs_probs.sum(axis=0)
            valid_indices = new_obs_probs_sum > 0
            new_obs_probs[:, valid_indices] /= new_obs_probs_sum[valid_indices]
            
            self.obs = new_obs_probs
            
            alpha_1 = self.forward(self.init, observations[0])
            beta_1 = betas[0, :]
            log_likelihood = np.log(np.sum(alpha_1 * beta_1))
            log_likelihoods.append(log_likelihood)
            
            if len(log_likelihoods) > 1:
                if np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < convergence_threshold:
                    has_converged = True
                    
        return self.obs, log_likelihoods
        
        
        
        
        