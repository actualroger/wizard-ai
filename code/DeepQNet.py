
from collections import OrderedDict
import torch.nn as nn

# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)

class DeepQNet(nn.Module):
    def __init__(self, params):
        super(DeepQNet, self).__init__()

        input_dim = params['observation_dim']
        num_hidden_layer = params['hidden_layer_num']
        dim_hidden_layer = params['hidden_layer_dim']
        output_dim = params['action_dim']

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, dim_hidden_layer)
        layers['inputRelu'] = nn.ReLU()
        for i in range(num_hidden_layer):
            layers['hidden%d' % i] = nn.Linear(dim_hidden_layer, dim_hidden_layer)
            layers['hidden%dRelu' % i] = nn.ReLU()
        layers['output'] = nn.Linear(dim_hidden_layer, output_dim)

        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        return self.linear_relu_stack(x)

class ConvDeepQNet(nn.Module):
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

# TODO dueling network
