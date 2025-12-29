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
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

model = VAE(input_dim=3*128*128, hidden_dim=1024, latent_dim=512).to(device)
epochs = 50
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss(reduction="sum")

for e in tqdm.tqdm(range(epochs)):
    losses = []
    for X, _ in dataloader:
        optim.zero_grad()
        X = X.to(device)

        out, mu, l = model(X)
        loss = criterion(out.view(-1), X.view(-1)) + (-0.5 * torch.sum(1 + l - mu.pow(2) - l.exp()))
        losses.append(loss.item())
        loss.backward()

        optim.step()

    print(f"epoch {e}, loss: {sum(losses) / len(losses)}")

torch.save(model.state_dict(), "vae.pth")