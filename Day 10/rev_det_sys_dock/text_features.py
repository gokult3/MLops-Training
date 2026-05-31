from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

df = pd.read_csv("reviews.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = df["review_text"].astype(str).tolist()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

np.save(
    "text_embeddings.npy",
    embeddings
)

print("Shape:", embeddings.shape)