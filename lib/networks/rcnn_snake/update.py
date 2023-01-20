import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.gru = ConvGRU(hidden_dim=64, input_dim=64)

        self.prediction = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 2, 1)
        )

    def forward(self, net, i_poly_fea):
        net = self.gru(net, i_poly_fea)
        offset = self.prediction(net).permute(0, 2, 1)
        return net, offset
