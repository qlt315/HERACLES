import torch
import numpy as np
from collections import deque
from rainbow_dqn.rainbow_sum_tree import SumTree


class ReplayBuffer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.current_size = 0
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.buffer['state'][self.count] = state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['next_state'][self.count] = next_state
        self.buffer['terminal'][self.count] = terminal
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # Convert numpy arrays to tensors
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None


class N_Steps_ReplayBuffer(object):
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.current_size = 0
        self.count = 0
        self.n_steps = args.n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]
        next_state, terminal = self.n_steps_deque[-1][3:5]
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def sample(self, total_steps):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # Convert numpy arrays to tensors
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None


class Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.current_size = 0
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.buffer['state'][self.count] = state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['next_state'][self.count] = next_state
        self.buffer['terminal'][self.count] = terminal
        # For the first experience, initialize priority to 1.0; for new experiences, assign the current maximum priority
        priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
        self.sum_tree.update(data_index=self.count, priority=priority)  # Update the priority of the current experience in sum_tree
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta: beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # Convert numpy arrays to tensors
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32)

        return batch, batch_index, IS_weight

    def update_batch_priorities(self, batch_index, td_errors):  # Update the priorities of the data at batch_index based on the given td_errors
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)


class N_Steps_Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = args.n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }
        self.current_size = 0
        self.count = 0

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            # For the first experience in the buffer, assign priority as 1.0; for new experiences, assign the current maximum priority
            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count, priority=priority)  # Update the priority of the current experience in sum_tree
            self.count = (self.count + 1) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta: beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # Convert numpy arrays to tensors
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32)

        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]  # Retrieve the state and action of the first transition in the deque
        next_state, terminal = self.n_steps_deque[-1][3:5]  # Retrieve the next state and terminal of the last transition in the deque
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # Calculate the n-steps reward in reverse order
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:  # If done=True, it indicates the end of an episode, so save the current transition's next state and terminal
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def update_batch_priorities(self, batch_index, td_errors):  # Update the priorities of the data at batch_index based on the given td_errors
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)
