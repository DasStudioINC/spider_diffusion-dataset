import torch

import torch.nn as nn

import torch.nn.functional as F



class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim



    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)

        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings



class ConditionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.cond_mlp = nn.Linear(time_dim, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)



    def forward(self, x, cond_emb):

        h = F.relu(self.bn1(self.conv1(x)))

        cond_features = self.cond_mlp(cond_emb)[..., None, None]

        h = h + cond_features

        h = F.relu(self.bn2(self.conv2(h)))

        return h



class SimpleConditionalUNet(nn.Module):

    def __init__(self, in_channels=3, base_dims=64, time_dim=128):

        super().__init__()

        self.time_mlp = nn.Sequential(

            SinusoidalPositionEmbeddings(time_dim),

            nn.Linear(time_dim, time_dim),

            nn.ReLU()

        )

        # Project CLIP text embeddings (512-dim) down to match time_dim

        self.text_projection = nn.Sequential(

            nn.Linear(512, time_dim),

            nn.ReLU(),

            nn.Linear(time_dim, time_dim)

        )



        self.init_conv = nn.Conv2d(in_channels, base_dims, 3, padding=1)

       

        # Expanded layers for 64x64 support (Double down/up sampling blocks)

        self.block1 = ConditionalBlock(base_dims, base_dims, time_dim)

        self.down1 = nn.Conv2d(base_dims, base_dims * 2, 4, stride=2, padding=1)



        self.block2 = ConditionalBlock(base_dims * 2, base_dims * 2, time_dim)

        self.down2 = nn.Conv2d(base_dims * 2, base_dims * 4, 4, stride=2, padding=1)



        self.block3 = ConditionalBlock(base_dims * 4, base_dims * 4, time_dim)



        self.up1 = nn.ConvTranspose2d(base_dims * 4, base_dims * 2, 4, stride=2, padding=1)

        self.block4 = ConditionalBlock(base_dims * 4, base_dims * 2, time_dim) # Concatenated with down2 features



        self.up2 = nn.ConvTranspose2d(base_dims * 2, base_dims, 4, stride=2, padding=1)

        self.block5 = ConditionalBlock(base_dims * 2, base_dims, time_dim) # Concatenated with down1 features



        self.out_conv = nn.Conv2d(base_dims, in_channels, 3, padding=1)



    def forward(self, x, t, text_embeddings):

        t_emb = self.time_mlp(t)

        text_emb = self.text_projection(text_embeddings)

        combined_cond = t_emb + text_emb



        x1 = F.relu(self.init_conv(x))

        h1 = self.block1(x1, combined_cond)

       

        x_down1 = self.down1(h1)

        h2 = self.block2(x_down1, combined_cond)

       

        x_down2 = self.down2(h2)

        h3 = self.block3(x_down2, combined_cond)



        # Upsampling path with skip connections

        x_up1 = self.up2_feat = self.up1(h3)

        x_up1 = torch.cat([x_up1, h2], dim=1)

        h4 = self.block4(x_up1, combined_cond)



        x_up2 = self.up2(h4)

        x_up2 = torch.cat([x_up2, h1], dim=1)

        h5 = self.block5(x_up2, combined_cond)



        return self.out_conv(h5)



def get_diffusion_schedule(timesteps=1000):

    beta_start = 0.0001

    beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, timesteps)

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return betas, alphas, alphas_cumprod



@torch.no_grad()

def sample_reverse_diffusion(model, image_shape, text_embeddings, timesteps=1000):

    device = next(model.parameters()).device

    betas, alphas, alphas_cumprod = get_diffusion_schedule(timesteps)

    betas = betas.to(device)

    alphas = alphas.to(device)

    alphas_cumprod = alphas_cumprod.to(device)



    img = torch.randn((1, *image_shape), device=device)

    model.eval()



    for i in reversed(range(timesteps)):

        t = torch.full((1,), i, dtype=torch.long, device=device)

        predicted_noise = model(img, t, text_embeddings)



        alpha = alphas[i]

        a_bar = alphas_cumprod[i]

        beta = betas[i]



        if i > 0:

            noise = torch.randn_like(img)

        else:

            noise = torch.zeros_like(img)



        img = (1.0 / torch.sqrt(alpha)) * (

            img - ((1.0 - alpha) / torch.sqrt(1.0 - a_bar)) * predicted_noise

        ) + torch.sqrt(beta) * noise



    return torch.clamp(img, -1.0, 1.0)