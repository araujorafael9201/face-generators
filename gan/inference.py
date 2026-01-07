import matplotlib.pyplot as plt
from models import Generator
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

model = Generator(3*128*128, 512).to(device)
model.load_state_dict(torch.load("gan/models/generator.pth"))

X = torch.randn(10, 128*128*3, device=device)
out = model.output(model(X))
out = out.view(X.size(0), 3, 128, 128).detach().cpu()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
	img = out[i].permute(1, 2, 0)
	ax.imshow(img)
	ax.axis('off')
plt.tight_layout()
plt.savefig("images/gan_results.png")
