import gradio as gr
import torch

from gan.models import Generator
from vae.model import VAE

LATENT_DIM_VAE = 128
INPUT_DIM = 3 * 128 * 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_models():
    vae = VAE(input_dim=INPUT_DIM, hidden_dim=256, latent_dim=LATENT_DIM_VAE).to(DEVICE)
    vae.load_state_dict(torch.load("vae/models/vae.pth", map_location=DEVICE))
    vae.eval()

    gan = Generator(INPUT_DIM, 512).to(DEVICE)
    gan.load_state_dict(torch.load("gan/models/generator.pth", map_location=DEVICE))
    gan.eval()

    return vae, gan

VAE_MODEL, GAN_MODEL = _load_models()

def generate(model_choice):
    with torch.no_grad():
        if model_choice == "GAN":
            latent = torch.randn(1, INPUT_DIM, device=DEVICE)
            output = GAN_MODEL.output(GAN_MODEL(latent))
        else:
            latent = torch.randn(1, LATENT_DIM_VAE, device=DEVICE)
            decoded = VAE_MODEL.output(VAE_MODEL.decoder(latent))
            output = decoded

    image = output.view(1, 3, 128, 128).cpu().squeeze(0).detach().numpy()
    image = (image * 255).astype("uint8").transpose(1, 2, 0)
    return image


demo = gr.Interface(
    fn=generate,
    inputs=gr.Radio(["VAE", "GAN"], label="Model"),
    outputs=gr.Image(label="Generated Face"),
)

demo.launch()