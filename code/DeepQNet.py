
from collections import OrderedDict
import torch.nn as nn
from torch import split, cat

# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the convolutional layer
    if isinstance(m, nn.Conv1d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)

class Qnet(nn.Module):
    # constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DeepQNet(Qnet):
    def __init__(self, params):
        super(DeepQNet, self).__init__()

        input_dim = params['observation_dim']
        num_hidden_layer = params['hidden_layer_num']
        dim_hidden_layer = params['hidden_layer_dim']
        output_dim = params['action_dim']

        layers = OrderedDict()
        layers['flatten'] = nn.Flatten(1, -1)
        layers['input'] = nn.Linear(input_dim, dim_hidden_layer)
        layers['inputRelu'] = nn.ReLU()
        for i in range(num_hidden_layer):
            layers['hidden%d' % i] = nn.Linear(dim_hidden_layer, dim_hidden_layer)
            layers['hidden%dRelu' % i] = nn.ReLU()
        layers['output'] = nn.Linear(dim_hidden_layer, output_dim)

        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        return self.linear_relu_stack(x)

class ConvDeepQNet(Qnet):
    def __init__(self, params):
        super(ConvDeepQNet, self).__init__()

        input_dim = params['observation_dim']
        conv_layer_num = params['conv_layer_num']
        conv_channel_num = params['conv_channel_num']
        conv_kernel_size = params['conv_kernel_size']
        num_hidden_layer = params['hidden_layer_num']
        dim_hidden_layer = params['hidden_layer_dim']
        output_dim = params['action_dim']

        layers = OrderedDict()
        layers['conv0'] = nn.Conv1d(1, conv_channel_num, conv_kernel_size)
        layers['convRelu0'] = nn.ReLU()
        for i in range(1, conv_layer_num-1):
            layers['conv%d' % i] = nn.Conv1d(conv_channel_num, conv_channel_num, conv_kernel_size)
            layers['conv%dRelu' % i] = nn.ReLU()
        layers['conv%d' % (conv_layer_num-1)] = nn.Conv1d(conv_channel_num, 1, conv_kernel_size)
        layers['conv%dRelu' % (conv_layer_num-1)] = nn.ReLU()
        layers['flatten'] = nn.Flatten(1, -1)
        layers['hidden0'] = nn.Linear(input_dim - conv_layer_num * (conv_kernel_size - 1), dim_hidden_layer)
        layers['hidden0Relu'] = nn.ReLU()
        for i in range(1, num_hidden_layer):
            layers['hidden%d' % i] = nn.Linear(dim_hidden_layer, dim_hidden_layer)
            layers['hidden%dRelu' % i] = nn.ReLU()
        layers['output'] = nn.Linear(dim_hidden_layer, output_dim)

        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        return self.linear_relu_stack(x)

class DuelingQNet(Qnet):
    def __init__(self, params):
        super().__init__()

        input_dim = params['observation_dim']
        conv_layer_num = params['conv_layer_num']
        conv_channel_num = params['conv_channel_num']
        conv_kernel_size = params['conv_kernel_size']
        value_num_hidden_layer = params['value_hidden_layer_num']
        value_dim_hidden_layer = params['value_hidden_layer_dim']
        action_num_hidden_layer = params['action_hidden_layer_num']
        action_dim_hidden_layer = params['action_hidden_layer_dim']
        output_dim = params['action_dim']

        convLayers = OrderedDict()
        convLayers['conv0'] = nn.Conv1d(1, conv_channel_num, conv_kernel_size)
        convLayers['convRelu0'] = nn.ReLU()
        for i in range(1, conv_layer_num-1):
            convLayers['conv%d' % i] = nn.Conv1d(conv_channel_num, conv_channel_num, conv_kernel_size)
            convLayers['conv%dRelu' % i] = nn.ReLU()
        convLayers['conv%d' % (conv_layer_num-1)] = nn.Conv1d(conv_channel_num, 1, conv_kernel_size)
        convLayers['conv%dRelu' % (conv_layer_num-1)] = nn.ReLU()
        convLayers['flatten'] = nn.Flatten(1, -1)

        valueLayers = OrderedDict()
        valueLayers['valueHidden0'] = nn.Linear(input_dim - conv_layer_num * (conv_kernel_size - 1), value_dim_hidden_layer)
        valueLayers['valueHidden0Relu'] = nn.ReLU()
        for i in range(1, value_num_hidden_layer):
            valueLayers['valueHidden%d' % i] = nn.Linear(value_dim_hidden_layer, value_dim_hidden_layer)
            valueLayers['valueHidden%dRelu' % i] = nn.ReLU()
        valueLayers['valueOutput'] = nn.Linear(value_dim_hidden_layer, 1)

        actionLayers = OrderedDict()
        actionLayers['actionHidden0'] = nn.Linear(input_dim - conv_layer_num * (conv_kernel_size - 1), action_dim_hidden_layer)
        actionLayers['actionHidden0Relu'] = nn.ReLU()
        for i in range(1, action_num_hidden_layer):
            actionLayers['actionHidden%d' % i] = nn.Linear(action_dim_hidden_layer, action_dim_hidden_layer)
            actionLayers['actionHidden%dRelu' % i] = nn.ReLU()
        actionLayers['actionOutput'] = nn.Linear(action_dim_hidden_layer, output_dim)

        self.conv_stack = nn.Sequential(convLayers)
        self.value_stack = nn.Sequential(valueLayers)
        self.action_stack = nn.Sequential(actionLayers)

    def forward(self, x):
        convolution = self.conv_stack(x)
        advantages = self.action_stack(convolution)
        return self.value_stack(convolution) - advantages.sum() + advantages

# generalized version below, composed of an input stack with parallel fully connected and convolutional layers
# then a dueling set of fully connected layers

# stack of convolutional layers
def ConvStack(layer_num, channel_num, kernel_size) -> nn.Module:
    assert(layer_num > 0)
    convLayers = OrderedDict()
    for i in range(layer_num):
        convLayers['conv%d' % i] = nn.Conv1d(1 if i == 0 else channel_num,
                                             1 if i == layer_num - 1 else channel_num,
                                             kernel_size)
        convLayers['conv%dRelu' % i] = nn.ReLU()
    return nn.Sequential(convLayers)

# stack of fully connected layers
def FullStack(input_dim, layer_num, layer_dim, output_dim, finalRelu: bool = False) -> nn.Module:
    assert(layer_num > 0)
    fullLayers = OrderedDict()
    for i in range(layer_num):
        fullLayers['full%d' % i] = nn.Linear(input_dim if i == 0 else layer_dim,
                                             output_dim if i == layer_num - 1 else layer_dim)
        if i < layer_num - 1 or finalRelu:
            fullLayers['full%dRelu' % i] = nn.ReLU()
    return nn.Sequential(fullLayers)

class GeneralDuelingQNet(Qnet):
    # constructor
    def __init__(self, params):
        super().__init__()

        observation_dim = params['observation_dim']
        input_header_len = params['input_header_len']
        input_full_layer_num = params['input_full_layer_num']
        input_full_layer_dim = params['input_full_layer_dim']
        conv_layer_num = params['conv_layer_num']
        conv_channel_num = params['conv_channel_num']
        conv_kernel_size = params['conv_kernel_size']
        value_num_hidden_layer = params['value_hidden_layer_num']
        value_dim_hidden_layer = params['value_hidden_layer_dim']
        action_num_hidden_layer = params['action_hidden_layer_num']
        action_dim_hidden_layer = params['action_hidden_layer_dim']
        action_dim = params['action_dim']

        self.input_header_len = input_header_len
        self.inputFullStack = FullStack(input_header_len, input_full_layer_num, input_full_layer_dim, input_full_layer_dim, True)
        self.input_cov_len = observation_dim - input_header_len
        assert(self.input_cov_len >= 0)
        self.inputConvStack = ConvStack(conv_layer_num, conv_channel_num, conv_kernel_size)
        self.intermediateSize = observation_dim - input_header_len - conv_layer_num * (conv_kernel_size - 1) + input_full_layer_dim
        self.flatten = nn.Flatten(1, -1)
        self.valueFullStack = FullStack(self.intermediateSize, value_num_hidden_layer, value_dim_hidden_layer, 1)
        self.actionFullStack = FullStack(self.intermediateSize, action_num_hidden_layer, action_dim_hidden_layer, action_dim)

    def forward(self, x):
        inputSplit = split(x, [self.input_header_len, self.input_cov_len], -1) # split data into header and deck
        inputFullProc = self.inputFullStack(inputSplit[0]) # full stack process header
        inputConvProc = self.inputConvStack(inputSplit[1]) # conv process deck
        intermediate = self.flatten(cat([inputFullProc, inputConvProc], -1)) # concatenate
        advantages = self.actionFullStack(intermediate) # evaluate actions
        return self.valueFullStack(intermediate) - advantages.sum() + advantages # add to values and normalize
