import numpy as np
import torch

class SumTree(object):
    """
    Stores data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storage:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # Capacity of the buffer
        self.tree_capacity = 2 * buffer_capacity - 1  # Capacity of the sum tree
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index, priority):
        # data_index represents the index of the current data in the buffer
        # tree_index represents the index of the current data in the sum tree
        tree_index = data_index + self.buffer_capacity - 1  # Convert the buffer index to the sum tree index
        change = priority - self.tree[tree_index]  # The change in priority for the current data
        self.tree[tree_index] = priority  # Update the priority of the leaf node at the bottom of the tree
        # Then propagate the change through the tree
        while tree_index != 0:  # Update the priority of the parent nodes, propagate the change to the top
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v):
        parent_idx = 0  # Start from the top of the tree
        while True:
            child_left_idx = 2 * parent_idx + 1  # Index of the left child node
            child_right_idx = child_left_idx + 1  # Index of the right child node
            if child_left_idx >= self.tree_capacity:  # Reached the bottom, end search
                tree_index = parent_idx  # The index of the sampled data in the sum tree
                break
            else:  # Search downwards, always look for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # Convert the tree index back to the buffer index
        return data_index, self.tree[tree_index]  # Return the index of the sampled data in the buffer and its priority

    def get_batch_index(self, current_size, batch_size, beta):
        batch_index = np.zeros(batch_size, dtype=np.long)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # Divide the range [0, priority_sum] into batch_size segments, and sample a number from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.get_index(v)
            batch_index[i] = index
            prob = priority / self.priority_sum  # The probability of the current data being sampled
            IS_weight[i] = (current_size * prob) ** (-beta)
        IS_weight /= IS_weight.max()  # Normalize

        return batch_index, IS_weight

    @property
    def priority_sum(self):
        return self.tree[0]  # The top of the tree holds the sum of all priorities

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()  # The leaf nodes store the priority of each data point
