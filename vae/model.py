import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        return self.hidden(X)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X):
        return self.hidden(X)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.l = nn.Linear(hidden_dim, latent_dim)

        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.output = nn.Sigmoid()

    def __reparameterize(self, mean, l):
      eps = torch.randn_like(l)
      return mean + eps * l

    def forward(self, X):
        X = self.flatten(X)
        X = self.encoder(X)
        mu = self.mu(X)
        l = self.l(X)
        X = self.decoder(self.__reparameterize(mu, torch.exp(0.5 * l)))
        out = self.output(X)

        return out, mu, l