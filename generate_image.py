import argparse
import re
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusion_core import SimpleConditionalUNet, sample_reverse_diffusion

def generate():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using your diffusion model.")
    parser.add_argument("--prompt", type=str, default="a cow riding a horse", help="The text prompt to guide image generation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using inference device: {device}")

    prompt = args.prompt
    print(f"Encoding text prompt: '{prompt}'...")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()

    text_inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(**text_inputs).pooler_output

    # Matches base_dims=64 from the training script
    model = SimpleConditionalUNet(in_channels=3, base_dims=64, time_dim=128).to(device)
    
    try:
        model.load_state_dict(torch.load("diffusion_checkpoint.pth", map_location=device))
        print("Loaded 'diffusion_checkpoint.pth' successfully!")
    except FileNotFoundError:
        print("Error: 'diffusion_checkpoint.pth' not found. Run training first.")
        return

    image_shape = (3, 64, 64)
    timesteps = 1000

    print("Running text-conditioned reverse sampling loop...")
    generated_tensor = sample_reverse_diffusion(model, image_shape, text_embeddings, timesteps=timesteps)

    # Post-process output tensor to image
    img_tensor = generated_tensor.squeeze(0).cpu()
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img_tensor = (img_tensor * 255).byte().permute(1, 2, 0).numpy()

    output_image = Image.fromarray(img_tensor)

    # Sanitize and truncate filename
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt).strip('_')
    safe_name = safe_name[:20] if len(safe_name) > 0 else "generated"
    output_filename = f"{safe_name}.png"

    output_image.save(output_filename)
    print(f"Generation complete! Saved result to '{output_filename}'.")

if __name__ == "__main__":
    generate()