from usearch.index import Index
from sentence_transformers import SentenceTransformer

with open("db6.txt", "r") as f:
    sentences = f.read().splitlines()

index = Index.restore('db6.usearch', view=True) 

MODEL_NAME = "all-mpnet-base-v2"  # Fast & lightweight; swap for a larger model if needed
model = SentenceTransformer(MODEL_NAME)

search = "Why is the sky blue?"
vector = model.encode(search)

matches = index.search(vector, 3)

for match in matches:
    i = match.key
    print(f"found {i}: {sentences[i]}")

