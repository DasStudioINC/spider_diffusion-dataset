import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusion_core import SimpleConditionalUNet, get_diffusion_schedule
from git_hub_imp.GitHubStreamDataset import GitHubStreamDataset

def train_model():
    parser = argparse.ArgumentParser(description="Train text-conditioned diffusion model via GitHub streaming dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")

    # 1. Load CLIP Text Encoder for Prompt Conditioning
    print("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()

    # 2. Setup GitHub Online Streaming Dataset
    # PASTE YOUR RAW GITHUB URL FOR dataset.json HERE:
    manifest_file = "https://raw.githubusercontent.com/DasStudioINC/spider_diffusion-dataset/main/json/dataset.json"

    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Matches 64x64 U-Net architecture
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1]
    ])

    train_dataset = GitHubStreamDataset(manifest_path=manifest_file, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=2) # Batch size of 2 since you have 4 images

    timesteps = 1000
    betas, alphas, alphas_cumprod = get_diffusion_schedule(timesteps)
    alphas_cumprod = alphas_cumprod.to(device)

    model = SimpleConditionalUNet(in_channels=3, base_dims=32, time_dim=128).to(device)

    # --- CHECKPOINT RESUMING LOGIC ---
    checkpoint_path = "diffusion_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from existing checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No existing checkpoint found. Starting training from scratch.")
    # ---------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    epochs = args.epochs
    print(f"\nStarting GitHub-streamed training loop for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_images, batch_prompts in dataloader:
            x_zero = batch_images.to(device)
            current_batch_size = x_zero.shape[0]

            # Encode the specific text prompts pulled live for this batch from GitHub
            with torch.no_grad():
                text_inputs = tokenizer(list(batch_prompts), padding=True, return_tensors="pt").to(device)
                batch_text_embeds = text_encoder(**text_inputs).pooler_output # Shape: [batch_size, 512]

            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            noise = torch.randn_like(x_zero)

            a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(a_bar) * x_zero + torch.sqrt(1.0 - a_bar) * noise

            predicted_noise = model(x_noisy, t, batch_text_embeds)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

    # Save final model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete! Final checkpoint saved to '{checkpoint_path}'.")

if __name__ == "__main__":
    train_model()