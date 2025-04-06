
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
    def __init__(self, input_dim, num_hidden_layer, dim_hidden_layer, output_dim):
        super(DeepQNet, self).__init__()

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
