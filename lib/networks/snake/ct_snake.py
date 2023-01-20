import torch.nn as nn
import torch

from .dla import DLASeg
from .ICD import RAFT


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.raft = RAFT()

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        output = self.raft(output, cnn_feature, batch)
        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
