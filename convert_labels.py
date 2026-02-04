import pandas as pd

# Read the CSV with text labels
df = pd.read_csv("english_fake_news_2212.csv")

# Convert text labels to numeric: Real=0, Fake=1
df['label'] = df['label'].map({'Real': 0, 'Fake': 1})

# Save the converted file
df.to_csv("english_fake_news_2212_numeric.csv", index=False)

print("✅ Conversion complete!")
print(f"Total rows: {len(df)}")
print(f"Real (0) count: {(df['label'] == 0).sum()}")
print(f"Fake (1) count: {(df['label'] == 1).sum()}")
