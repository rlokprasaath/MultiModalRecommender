import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# === Paths ===
INPUT_CSV = './data/cleaned_text.csv'
OUTPUT_FEATURES = './embeddings/text_features.npy'
OUTPUT_IDS = './embeddings/text_ids.txt'

# === Load preprocessed text ===
df = pd.read_csv(INPUT_CSV)
print(f"📄 Loaded {len(df)} rows from {INPUT_CSV}")

# === TF-IDF vectorizer ===
vectorizer = TfidfVectorizer(
    max_features=1000,    # limit to 1000 features for simplicity
    stop_words='english'
)

print("🔄 Extracting TF-IDF features...")
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# === Save features as .npy ===
text_features = tfidf_matrix.toarray()
np.save(OUTPUT_FEATURES, text_features)

# === Save the corresponding IDs ===
with open(OUTPUT_IDS, 'w') as f:
    for id in df['id']:
        f.write(f"{id}\n")

print(f"✅ Saved TF-IDF features to: {OUTPUT_FEATURES}")
print(f"🧾 Saved text IDs to: {OUTPUT_IDS}")
print(f"📐 Shape of features: {text_features.shape}")
