import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class VQAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VQAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.hypernet_w = nn.Linear(args.role_dim, args.rnn_hidden_dim * args.n_actions)
        # self.hypernet_b = nn.Linear(args.role_dim, args.n_actions)
        self.hypernet_w = nn.Sequential(nn.Linear(args.role_dim, args.rnn_hidden_dim * args.n_actions // 2), nn.ReLU(),
                                        nn.Linear(args.rnn_hidden_dim * args.n_actions // 2, args.rnn_hidden_dim * args.n_actions))
        self.hypernet_b = nn.Sequential(nn.Linear(args.role_dim, args.rnn_hidden_dim * args.n_actions // 2), nn.ReLU(),
                                        nn.Linear(args.rnn_hidden_dim * args.n_actions // 2, args.n_actions))
        
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.hypernet_w, gain=args.gain)
            orthogonal_init_(self.hypernet_b, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, role_embed):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        if getattr(self.args, "use_layer_norm", False):
            h = self.layer_norm(h)

        w = self.hypernet_w(role_embed.detach()).view(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        b = self.hypernet_b(role_embed.detach()) 
        q = th.bmm(h.unsqueeze(1), w).squeeze(1) + b
        return q, h    