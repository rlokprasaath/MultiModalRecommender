import numpy as np

# Load IDs
with open('./embeddings/image_ids.txt') as f:
    image_ids = [line.strip() for line in f]

with open('./embeddings/text_ids.txt') as f:
    text_ids = [line.strip() for line in f]

# Load features
image_features = np.load('./embeddings/image_features.npy')
text_features = np.load('./embeddings/text_features.npy')

# Find common IDs
common_ids = list(set(image_ids) & set(text_ids))
common_ids.sort()

# Create mapping from ID → index
image_id_to_index = {id_: i for i, id_ in enumerate(image_ids)}
text_id_to_index = {id_: i for i, id_ in enumerate(text_ids)}

# Align features
aligned_image_features = np.array([image_features[image_id_to_index[id_]] for id_ in common_ids])
aligned_text_features = np.array([text_features[text_id_to_index[id_]] for id_ in common_ids])

# Combine
combined = np.concatenate([aligned_image_features, aligned_text_features], axis=1)

# Save
np.save('./embeddings/combined_features.npy', combined)

# Save aligned common IDs
with open('./embeddings/combined_ids.txt', 'w') as f:
    for id_ in common_ids:
        f.write(f"{id_}\n")

print(f"Combined shape: {combined.shape}")
print(f"Common aligned IDs saved to: combined_ids.txt")
