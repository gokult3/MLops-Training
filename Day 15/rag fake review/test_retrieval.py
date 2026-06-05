import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

index = faiss.read_index("reviews.index")

metadata = pd.read_csv("reviews_metadata.csv")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

query = "Best product ever"

query_vec = embedder.encode([query])

distances, indices = index.search(
    np.array(query_vec),
    3
)

for idx in indices[0]:
    print(metadata.iloc[idx]["review_text"])
import google.generativeai as genai

genai.configure(
    api_key="YOUR_API_KEY"
)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Explain why this review looks fake"
)

print(response.text)