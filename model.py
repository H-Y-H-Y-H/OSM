from torch import nn, optim
import torch


class FastNN(nn.Module):
    def __init__(self, s, a):
        super(FastNN, self).__init__()
        self.num_layers = 1
        self.hidden_size = 128

        # self.l1 = nn.LSTM(s*3 + a, self.hidden_size, self.num_layers, batch_first = True)
        self.l1 = nn.Linear(s * 3 + a, 128)
        self.l2 = nn.Linear(128, 128)
        # self.l3 = nn.LSTM(128, self.hidden_size, self.num_layers, batch_first = True)
        self.l3 = nn.Linear(128, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, s)
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(0.8)

    def reset(self, s):
        pass

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def forward(self, s, a):
        # x1,(h1,c1) = self.l1(s[0]).unsqueeze(0),(h0,c0))
        # x1 = self.tanh(x1)

        # x2,(h2,c2) = self.l2(s[1]).unsqueeze(0),(h1,c1))
        # x2 = self.tanh(x2)
        # print(s.shape,a.shape)
        x = torch.cat([s, a], -1).unsqueeze(0)
        # x, (h1,c1) = self.l1(x, None)

        x = self.tanh(self.l1(x))
        x = self.tanh(self.l2(x))

        # x,_ = self.l3(x,(h1,c1))
        x = self.tanh(self.l3(x))
        # x = self.dropout(x)
        x = self.l4(x)
        x = self.l5(x)
        return x