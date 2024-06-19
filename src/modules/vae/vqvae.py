import torch as th 
import torch.nn as nn
import torch.nn.functional as F
from .nearest_embed import NearestEmbed, NearestEmbedEMA
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class VQVAE(nn.Module):
    def __init__(self, input_shape, state_dim, args):
        super(VQVAE, self).__init__()
        self.args = args
        self.state_dim = state_dim

        self.fc0 = nn.Linear(input_shape, args.vae_hidden_dim)
        self.fc1 = nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim)
        self.fc2 = nn.Linear(args.vae_hidden_dim, args.role_dim)

        self.emb = NearestEmbed(args.n_roles, args.role_dim)

        self.fc3 = nn.Linear(args.role_dim, args.vae_hidden_dim)
        self.fc4 = nn.Linear(args.n_agents * args.vae_hidden_dim, args.vae_hidden_dim)
        self.fc5 = nn.Linear(args.vae_hidden_dim, state_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc0)
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2)
            orthogonal_init_(self.fc3)
            orthogonal_init_(self.fc4)
            orthogonal_init_(self.fc5)

    def encode(self, inputs):
        return self.fc2(F.relu(self.fc1(F.relu(self.fc0(inputs)))))
    
    def decode(self, z):
        rec = F.relu(self.fc3(z)).view(-1, self.args.n_agents * self.args.vae_hidden_dim)
        rec = self.fc5(F.relu(self.fc4(rec)))
        return rec

    def forward(self, inputs):
        z_e = self.encode(inputs)
        z_q, _ = self.emb(z_e, weight_sg=True)
        # preserve gradients
        # z_q = z_e + (z_q - z_e).detach()
        role_emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, role_emb

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.mse_loss(recon_x, x)
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.args.vq_coef * self.vq_loss + self.args.commit_coef * self.commit_loss