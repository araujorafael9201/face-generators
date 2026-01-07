import matplotlib.pyplot as plt
from model import VAE
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

latent_dim=512
model = VAE(input_dim=3*128*128, hidden_dim=1024, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("vae.pth"))

X = torch.randn(10, latent_dim, device=device)
out = model.output(model.decoder(X))
out = out.view(X.size(0), 3, 128, 128).detach().cpu()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
	img = out[i].permute(1, 2, 0)
	ax.imshow(img)
	ax.axis('off')
plt.tight_layout()
plt.show()
