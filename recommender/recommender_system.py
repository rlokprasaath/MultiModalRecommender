import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import os

# === Paths ===
COMBINED_FEATURES_PATH = './embeddings/combined_features.npy'
COMBINED_IDS_PATH = './embeddings/combined_ids.txt'
IMAGE_FOLDER = './data/images'  # 🔁 Make sure this matches your folder name

# === Load features and IDs ===
features = np.load(COMBINED_FEATURES_PATH)

with open(COMBINED_IDS_PATH) as f:
    ids = [line.strip() for line in f]

# === Function: Show images in grid ===
def show_images(image_ids, ncols=5):
    n = len(image_ids)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 3, nrows * 4))

    for i, id_ in enumerate(image_ids):
        img_path = os.path.join(IMAGE_FOLDER, f"{id_}.jpg")
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path)
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(image)
            plt.title(f"ID: {id_}", fontsize=10)
            plt.axis('off')
        except Exception as e:
            print(f"❌ Failed to load {img_path}: {e}")
            continue

    plt.tight_layout()
    plt.show()

# === Function: Get top N similar items ===
def recommend(product_id, top_n=5):
    if product_id not in ids:
        print("❌ Product ID not found.")
        return

    idx = ids.index(product_id)
    query_vector = features[idx].reshape(1, -1)

    similarities = cosine_similarity(query_vector, features)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    print(f"\n🔍 Top {top_n} recommendations for product ID: {product_id}\n")

    top_ids = []
    for i in ranked_indices[1:top_n+1]:  # skip self
        print(f"→ ID: {ids[i]} | Score: {similarities[i]:.4f}")
        top_ids.append(ids[i])

    print("\n🖼️ Showing recommended images:")
    show_images(top_ids)

# === Entry point ===
if __name__ == '__main__':
    sample_id = input("Enter product ID to recommend: ").strip()
    recommend(sample_id, top_n=5)
