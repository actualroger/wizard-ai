
import numpy as np

# defines a Buffer class which all other buffers should descend from
class Buffer(object):
    def __init__(self, size):
        self.total_size = size
        # create an rng object and link its seed to the global seed
        self.rng = np.random.default_rng(np.random.randint(1e9))

    def add(self, tuple):
        raise NotImplementedError("Buffer subclass must implement add(tuple)")

    def sampleBatch(self, batch_size):
        raise NotImplementedError("Buffer subclass must implement sample_batch(batch_size)")

# standard replay buffer built on circular array
class ReplayBuffer(Buffer):
    """ Implement the Replay Buffer as a class, which contains:
            - self._data_buffer (list): a list variable to store all transition tuples.
            - add: a function to add new transition tuple into the buffer
            - sample_batch: a function to sample a batch training data from the Replay Buffer
    """
    def __init__(self, buffer_size):
        """Args:
               buffer_size (int): size of the replay buffer
        """
        # total size of the replay buffer
        super().__init__(buffer_size)

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, actMask, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, actMask, reward, next_obs, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        """ Function to fetch the state, action, reward, next state, and done arrays.

            Args:
                indices (list): list contains the index of all sampled transition tuples.
        """
        # lists for transitions
        obs_list, actions_list, action_masks_list, rewards_list, next_obs_list, dones_list = [], [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, actMask, reward, next_obs, d = data
            # convert to np arrays
            obs_list.append(np.asarray(obs))
            actions_list.append(np.asarray(act))
            action_masks_list.append(np.asarray(actMask))
            rewards_list.append(np.asarray(reward))
            next_obs_list.append(np.asarray(next_obs))
            dones_list.append(np.asarray(d))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(action_masks_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sampleBatch(self, batch_size):
        """ Args:
                batch_size (int): size of the sampled batch data.
        """
        # sample indices with replacement
        indices = self.rng.choice(len(self._data_buffer), size=batch_size, replace=False, shuffle=False)
        return self._encode_sample(indices)

# replay buffer with additional priority list used to select items
class PriorityBuffer(ReplayBuffer):
    # constructor
    def __init__(self, buffer_size, params):
        super().__init__(buffer_size)
        self.params = params
        self._weights = []
        self._total_weight = 0
        self.prev_weights = [] # for reference during loss calculation

    # calculate and add weight
    def add(self, *args):
        super().add(*args) # this increments self._next_idx
        next_idx = self.__len__() - 1 if self._next_idx == 0 else self._next_idx - 1

        weight = self.priorityFunction(self._data_buffer[next_idx])

        # interesting implementation
        if next_idx >= len(self._weights):
            self._weights.append(weight)
            self._total_weight += weight
        else:
            self._total_weight += weight - self._weights[next_idx]
            self._weights[next_idx] = weight

    def sampleBatch(self, batch_size):
        """ Args:
                batch_size (int): size of the sampled batch data.
        """
        # sample indices with replacement
        indices = self.rng.choice(len(self._data_buffer),
                                  size=batch_size,
                                  p=[w / self._total_weight for w in self._weights],
                                  replace=False,
                                  shuffle=False)
        self.prev_weights = [self._weights[i] for i in indices]
        return self._encode_sample(indices)

    def priorityFunction(self, transition) -> float:
        raise NotImplementedError("Subclass of PriorityBuffer must implement priorityFunction(self, transition) -> float")

# replay buffer which prioritizes by action index
class ActionPriorityBuffer(PriorityBuffer):
    # constructor
    def __init__(self, *args):
        super().__init__(*args)

    # prioritize by action
    def priorityFunction(self, transition):
        action = transition[1]
        return (1.0 if action in range(60) else # normal card playment = 1
            (action - 56) * 0.5 if action in range(60, 81) else # betting = 2-10
            10 if action in range(81, 85) else # trump = 10
            1) # points = 1
