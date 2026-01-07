import torch
import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader

from models import Discriminator, Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

data_directory="data"
dataset = ImageFolder(data_directory, transform=Compose([Resize((128,128)), ToTensor()]))
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)

generator = Generator(3*128*128, 512).to(device)
discriminator = Discriminator(3*128*128, 512).to(device)

epochs = 50
k = 3
d_optim = torch.optim.AdamW(discriminator.parameters(), lr=1e-3)
g_optim = torch.optim.AdamW(generator.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss(reduction="sum")
checkpoint_steps = 5

for e in tqdm.tqdm(range(epochs), desc="Epochs"):
    for i, (X, _) in enumerate(tqdm.tqdm(dataloader, leave=False, desc=f"Epoch {e} progress")):
        d_optim.zero_grad()
        X = X.to(device)

        fake_X = generator(torch.randn_like(X, device=device))
        real_X = X

        fake_logits = discriminator(fake_X).squeeze()
        real_logits = discriminator(real_X).squeeze()

        d_loss = criterion(fake_logits, torch.zeros(X.size(0), device=device)) + criterion(real_logits, torch.ones(X.size(0), device=device))
        d_loss.backward()
        d_optim.step()

        if i % k == 0:
            g_optim.zero_grad()
            fake_X = generator(torch.randn_like(X, device=device))
            g_loss = criterion(discriminator(fake_X).squeeze(), torch.ones(X.size(0), device=device))
            g_loss.backward()
            g_optim.step()

    if (e+1) % checkpoint_steps == 0 or (e+1) == epochs:
        torch.save(discriminator.state_dict(), f"gan/models/discriminator.pth")
        torch.save(generator.state_dict(), f"gan/models/generator.pth")

    print(f"d_loss={d_loss.item()}\tg_loss={g_loss.item()}")