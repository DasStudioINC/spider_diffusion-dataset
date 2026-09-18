import argparse
import re
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusion_core import SimpleConditionalUNet, sample_reverse_diffusion

import os
import model as model_params

from git_hub_imp.GitHubStreamDataset import push_changes

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

    # Matches base_dims=128 and time_dim=256 from the upgraded training script
    model = SimpleConditionalUNet(in_channels=3, base_dims=model_params.gen_X_scale, time_dim=model_params.gen_Y_scale).to(device)
    
    try:
        model.load_state_dict(torch.load(model_params.model_name, map_location=device))
        print(f"Loaded '{model_params.model_name}' successfully!")
    except FileNotFoundError:
        print(f"Error: '{model_params.model_name}' not found. Run training first.")
        return

    image_shape = model_params.image_shape
    timesteps = model_params.gen_timesteps

    print("Running text-conditioned reverse sampling loop...")
    generated_tensor = sample_reverse_diffusion(model, image_shape, text_embeddings, timesteps=timesteps)

    # Post-process output tensor to image
    img_tensor = generated_tensor.squeeze(0).cpu()
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img_tensor = (img_tensor * 255).byte().permute(1, 2, 0).numpy()

    output_image = Image.fromarray(img_tensor)

    model_params.image_Gen_Count()
    # Sanitize and truncate filename
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt).strip('_')
    safe_name = safe_name[:20] if len(safe_name) > 0 else "generated"
    output_filename = f"{safe_name}#{model_params.image_Gen_Count()}.png"

    

    models_dir = "./model_storage"
    os.makedirs(models_dir, exist_ok=True)
    output_path = os.path.join(models_dir, output_filename)
    
    output_image.save(output_path)
    output_image.save(output_filename)
    #push_changes(repo_path=models_dir, commit_message=f"New Image Created @{output_filename}")
    print(f"Finished creating image located at {output_image}")

if __name__ == "__main__":
    generate()