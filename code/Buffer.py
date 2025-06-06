
import numpy as np
import Schedule

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

    def add(self, *args):
        # create a tuple
        trans = (args)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    # fetch data arrays
    def _encode_sample(self, indices):
        # lists for transitions
        numLists = len(self._data_buffer[indices[0]])
        outputList = [[] for _ in range(numLists)]

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            for l in range(numLists):
                outputList[l].append(np.asarray(data[l]))
        # return the sampled batch data as list of numpy arrays
        return [np.array(o) for o in outputList]

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
        normWeights = [w / self._total_weight for w in self._weights]
        # sample indices with replacement
        indices = self.rng.choice(len(self._data_buffer),
                                  size=batch_size,
                                  p=normWeights,
                                  replace=False,
                                  shuffle=False)
        is_weight = np.array([normWeights[i] for i in indices])
        self.prev_weights = is_weight / is_weight.min()
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

# replay buffer which prioritizes by predicted amount of Q loss remaining per experience
# aping https://github.com/rlcode/per/blob/master/cartpole_per.py
class QPriorityBuffer(PriorityBuffer):
    # constructor
    def __init__(self, buffer_size, params):
        super().__init__(buffer_size, params)

        # parameters
        self.error_floor = params['buffer_error_floor'] # error floor
        self.error_pow = params['buffer_error_pow'] # error power

        betaScheduleType = params['buffer_beta_schedule_type']
        if betaScheduleType == 'none':
            self.betaSchedule = None
        else:
            betaParams = {'schedule_type' : betaScheduleType,
                        'start_value' : params['buffer_beta_start_value'],
                        'end_value' : params['buffer_beta_end_value'],
                        'duration' : params['buffer_beta_duration']}
            self.betaSchedule = Schedule.createSchedule(betaParams)
            if self.betaSchedule is None:
                raise RuntimeError("No beta schedule provided")
        self.step = 0 # timestep for beta schedule

        self.prev_indices = [] # previous sample batch indices

    # add experience only
    def add(self, *args):
        # create a tuple
        trans = (args)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    # calculate and add weight (THIS MUST BE CALLED DIRECTLY AFTER .ADD())
    def addError(self, error):
        next_idx = self.__len__() - 1 if self._next_idx == 0 else self._next_idx - 1
        weight = self.priorityFunction(error)

        # interesting implementation
        if next_idx >= len(self._weights):
            self._weights.append(weight)
            self._total_weight += weight
        else:
            self._total_weight += weight - self._weights[next_idx]
            self._weights[next_idx] = weight

    # add weight by index
    def setErrors(self, indices, errors):
        for i in range(len(indices)):
            self._total_weight += errors[i].item() - self._weights[indices[i]]
            self._weights[indices[i]] = errors[i].item()

    # samples a batch and saves weights to self.prev_weights
    def sampleBatch(self, batch_size):
        normWeights = [w / self._total_weight for w in self._weights]
        # sample indices with replacement
        indices = self.rng.choice(len(self._data_buffer),
                                  size=batch_size,
                                  p=normWeights,
                                  replace=False,
                                  shuffle=False)
        
        # calculate sampling probabilities
        rawWeights = [normWeights[i] for i in indices]
        if self.betaSchedule is not None:
            is_weight = np.power(self.__len__() * rawWeights, -self.betaSchedule.getValue(self.step))
        else:
            is_weight = 1 / (self.__len__() * np.array(rawWeights))
        self.step += 1
        self.prev_weights = is_weight / is_weight.max()
        self.prev_indices = indices
        return self._encode_sample(indices)

    # prioritize by Q error
    def priorityFunction(self, error):
        return (abs(error) + self.error_floor) ** self.error_pow
