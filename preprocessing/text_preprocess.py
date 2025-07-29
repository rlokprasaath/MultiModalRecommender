import pandas as pd
import re
import os

# === Paths ===
INPUT_CSV = './data/styles.csv'
OUTPUT_CSV = './data/cleaned_text.csv'

# === Text Cleaning Function ===
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text

# === Load styles.csv (subset) ===
df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')

# === Fill missing columns (if any) ===
required_cols = ['gender', 'articleType', 'baseColour', 'productDisplayName']
for col in required_cols:
    if col not in df.columns:
        df[col] = ''

# === Combine and clean relevant columns ===
df['combined_text'] = (
    df['gender'].astype(str) + ' ' +
    df['articleType'].astype(str) + ' ' +
    df['baseColour'].astype(str) + ' ' +
    df['productDisplayName'].astype(str)
).apply(clean_text)

# === Keep only needed columns ===
output_df = df[['id', 'combined_text']]
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Cleaned text saved to: {OUTPUT_CSV}")
print(f"Total records: {len(output_df)}")
