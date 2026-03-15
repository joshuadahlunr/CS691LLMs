# pip install sentence-transformers nltk usearch

import sys
from sentence_transformers import SentenceTransformer
from usearch.index import Index
from itertools import chain
import nltk
import numpy as np

nltk.download('punkt')

# --- Config ---
input_file = "pairs.txt"
db_name = "pairs"
MODEL_NAME = "all-mpnet-base-v2"

# --- Load document from file ---
try:
    with open(input_file, "r", encoding="utf-8") as f:
        document = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# --- Step 1: Split into sentences ---
sentences = document.splitlines()
pairs = [[p.strip("\n\t\" ") for p in x.split("->")] for x in sentences]
# for p in pairs:
#     if len(p) == 1:
#         print(p)
sentences = [item for sub_list in pairs for item in sub_list]

# --- Step 2: Encode to vectors ---
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(sentences, show_progress_bar=True)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Each sentence → a {embeddings.shape[1]}-dimensional vector")

# --- Step 3: Build index ---
index = Index(ndim=embeddings.shape[1])

for i in range(len(sentences)):
    index.add(i, embeddings[i])

# --- Step 4: Save outputs ---
text_output = f"{db_name}.txt"
index_output = f"{db_name}.usearch"

with open(text_output, "w", encoding="utf-8") as f:
    f.write("\n".join([" -> ".join(pair) for pair in pairs]))

index.save(index_output)

print(f"\nSaved sentences to: {text_output}")
print(f"Saved index to: {index_output}")