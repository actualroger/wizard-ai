
import numpy as np

# defines a Buffer class which all other buffers should descend from
class Buffer(object):
    def __init__(self, size):
        self.total_size = size

    def add(self, tuple):
        raise NotImplementedError("Buffer subclass must implement add(tuple)")

    def sampleBatch(self, batch_size):
        raise NotImplementedError("Buffer subclass must implement sample_batch(batch_size)")

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
            # store to the list
            # obs_list.append(np.array(obs, copy=False))
            # actions_list.append(np.array(act, copy=False))
            # rewards_list.append(np.array(reward, copy=False))
            # next_obs_list.append(np.array(next_obs, copy=False))
            # dones_list.append(np.array(d, copy=False))
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
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)

# TODO implement priority buffer
# TODO implement deque buffer?
