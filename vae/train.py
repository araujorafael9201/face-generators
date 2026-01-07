import os
import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize

from model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

data_directory="data"
dataset = ImageFolder(data_directory, transform=Compose([Resize((128,128)), ToTensor()]))
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4)

model = VAE(input_dim=3*128*128, hidden_dim=256, latent_dim=128).to(device)
if os.path.exists("vae/models/vae.pth"):
    print("starting from checkpoint")
    model.load_state_dict(torch.load("vae/models/vae.pth", weights_only=True))
epochs = 50
lr = 1e-3
optim = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction="sum")
checkpoint_steps = 5
lr_factor = 0.1

for e in tqdm.tqdm(range(epochs), desc="Epochs"):
    losses = []
    for X, _ in tqdm.tqdm(dataloader, leave=False, desc=f"Epoch {e} progress"):
        optim.zero_grad()
        X = X.to(device)

        out, mu, l = model(X)
        loss = criterion(out.view(-1), X.view(-1)) + (-0.5 * torch.sum(1 + l - mu.pow(2) - l.exp()))
        losses.append(loss.item())
        loss.backward()

        optim.step()

    if (e + 1) % checkpoint_steps == 0 or (e + 1) == epochs:
        torch.save(model.state_dict(), f"vae/models/vae.pth")

    if (e + 1) % 20 == 0:
        lr *= lr_factor
        optim = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"epoch {e}, loss: {sum(losses) / len(losses)}")