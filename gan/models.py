import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, X):
        X = self.input(X)
        X = self.hidden(X)
        return self.output(X)
    
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        self.output = nn.Sigmoid()

    def forward(self, X):
        X = self.input(X)
        X = self.hidden(X)
        return self.output(X)