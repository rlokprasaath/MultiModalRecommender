import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

# === Paths ===
INPUT_FOLDER = './data/images'
OUTPUT_FOLDER = './embeddings/image_embeddings'

# === Make sure output directory exists ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Define image preprocessing transform (PyTorch-style) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Process each image ===
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')]
skipped = 0
processed = 0

print(f"Starting image preprocessing for {len(image_files)} files...")

for filename in tqdm(image_files, desc="Processing"):
    img_id = os.path.splitext(filename)[0]
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{img_id}.npy")

    try:
        image = Image.open(input_path).convert('RGB')
        image_tensor = transform(image).numpy()  # Shape: (3, 224, 224)
        np.save(output_path, image_tensor)
        processed += 1
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"Skipping {filename}: {e}")
        skipped += 1

print(f"\nDone! Processed: {processed} images | Skipped: {skipped}")
print(f"Saved to: {OUTPUT_FOLDER}")
