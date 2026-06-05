import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("reviews.csv")  # columns: review_text, label
df = df.dropna(subset=["review_text"])
df = df.head(5000)  # use 5000 reviews for speed

# Encode reviews
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["review_text"].tolist(), show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, "reviews.index")
df[["review_text", "label"]].to_csv("reviews_metadata.csv", index=False)

print("Index built successfully!")
import pandas as pd
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

df = pd.read_csv("reviews.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

reviews = df["review_text"].tolist()

embeddings = model.encode(
    reviews,
    show_progress_bar=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "reviews.index")

df.to_csv(
    "reviews_metadata.csv",
    index=False
)

print("FAISS Index Created")