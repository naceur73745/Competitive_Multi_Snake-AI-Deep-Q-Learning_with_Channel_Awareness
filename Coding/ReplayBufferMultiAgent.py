import numpy as np 

class ReplayBuffer:
    def __init__(self, capacity, input_dim, n_actions):
        self.capacity = capacity
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.mem_cntr = 0
        self.states = np.zeros((self.capacity, self.input_dim))
        self.next_states = np.zeros((self.capacity, self.input_dim))
        self.actions = np.zeros((self.capacity, self.n_actions))
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity, dtype=bool)

    def store_transition(self, state, next_state, action, reward, done):
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.mem_cntr += 1

    def sample_batch(self, batch_size):

        max_mem = min(self.mem_cntr, self.capacity)

        if max_mem >= batch_size:
             batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        else:
             batch_indices = np.random.choice(max_mem, batch_size, replace=True)


        states = self.states[batch_indices]
        next_states = self.next_states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones = self.dones[batch_indices]

        return states, next_states, actions, rewards, dones