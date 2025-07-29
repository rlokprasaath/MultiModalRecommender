import os
import numpy as np
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm

# === Paths ===
INPUT_FOLDER = '/content/multi_modal_recommender/embeddings/image_embeddings'
OUTPUT_FEATURES = '/content/multi_modal_recommender/embeddings/image_features.npy'
OUTPUT_IDS = '/content/multi_modal_recommender/embeddings/image_ids.txt'

# === Load ResNet50 (remove final layer) ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # remove classifier
resnet = resnet.to(device)
resnet.eval()

# === Process each .npy image file ===
image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.npy')])
features = []
image_ids = []

print(f"Extracting features from {len(image_files)} images...")

with torch.no_grad():
    for file in tqdm(image_files):
        img_id = os.path.splitext(file)[0]
        path = os.path.join(INPUT_FOLDER, file)
        
        try:
            image_tensor = np.load(path)
            image_tensor = torch.tensor(image_tensor).unsqueeze(0).to(device).float()  # shape: (1, 3, 224, 224)
            output = resnet(image_tensor)  # shape: (1, 2048)
            features.append(output.cpu().numpy().squeeze())
            image_ids.append(img_id)
        except Exception as e:
            print(f"Skipped {file}: {e}")

# Ensure the output folder exists
os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)

# === Save output ===
features_np = np.stack(features)
np.save(OUTPUT_FEATURES, features_np)

with open(OUTPUT_IDS, 'w') as f:
    for id in image_ids:
        f.write(f"{id}\n")

print("Done! Saved:")
print(f"image_features.npy → shape: {features_np.shape}")
print(f"image_ids.txt → {len(image_ids)} IDs")

# this was ran in colab for acheiving the speed of gpu