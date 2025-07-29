#  Multi-Modal Product Recommender System (Image + Text)

A hybrid content-based recommendation system that suggests similar fashion products based on both **visual appearance** and **textual metadata** (e.g., product category, brand, fabric). This system uses **ResNet50** for image feature extraction and **TF-IDF** for text, combining them for powerful recommendations.

---

##  Features

-  Extracts deep image features using pretrained **ResNet50**
-  Analyzes product metadata using **TF-IDF** vectorization
-  Combines both features to generate **multi-modal embeddings**
-  Finds and displays visually + semantically similar items
-  Works using just a **product ID** as input (no manual tags required)

---

##  Tech Stack

- **Python 3.10+**
- `TensorFlow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `Pillow`

---

##  Project Structure

multi_modal_recommender/
├── data/
│ ├── styles_subset.csv # Cleaned product metadata
│ └── images_subset/ # (Optional) 5–10 demo images
├── embeddings/
│ ├── image_features.npy # ResNet image vectors
│ ├── text_features.npy # TF-IDF vectors
│ ├── combined_features.npy # Combined image+text features
│ ├── image_ids.txt
│ ├── text_ids.txt
│ └── combined_ids.txt
├── preprocessing/
│ ├── image_preprocessing.py # Extract image features
│ └── text_preprocessing.py # Process & vectorize text
├── recommender/
│ ├── combine_features.py # Merge image and text features
│ └── recommender_system.py # Recommend similar products
├── requirements.txt
└── README.md

---

##  How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the recommender

python recommender/recommender_system.py

---

### Example output

<img src="sample_output.png" width="600" />

---
