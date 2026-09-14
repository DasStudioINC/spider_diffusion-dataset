import os 
import json
import requests
from PIL import Image
import torch
from torch.utils.data import IterableDataset

class GitHubStreamDataset(IterableDataset):
   def __init__(self, manifest_path, transform, cache_dir="./cache_images"):
        super().__init__()
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform
        
        # Check if manifest_path is a raw GitHub URL or a local file
        if manifest_path.startswith("http://") or manifest_path.startswith("https://"):
            print(f"Fetching dataset manifest from GitHub: {manifest_path}")
            response = requests.get(manifest_path, timeout=15)
            response.raise_for_status()
            self.data = response.json()
        else:
            with open(manifest_path, "r") as f:
                self.data = json.load(f)
            
        self.cache_dir = cache_dir
    
    def __iter__(self):
        for item in self.data:
            url = item["url"]
            prompt = item["prompt"]

            filename = os.path.basename(url)
            cached_path = os.path.join(self.cache_dir, filename)

            if not os.path.exists(cached_path):
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        with open(cached_path, "wb") as f:
                            f.write(response.content)
                    else:
                        print(f"Warning: Failed to fetch {url} (Status: {response.status_code})")
                        continue
                except Exception as e:
                    print(f"Network error downloading {url}: {e}")
                    continue

                try:
                    image = Image.open(cached_path).convert("RGB")
                    tensor_image = self.transform(image)
                    yield tensor_image, prompt
                except Exception as e:
                    print(f"Error processing image {cached_path}: {e}")
                    continue
