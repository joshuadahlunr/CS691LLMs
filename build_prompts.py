from usearch.index import Index
from sentence_transformers import SentenceTransformer

messages = ["<none>"] * 3

with open("db5.txt", "r") as f:
    sentences = f.read().splitlines()

index = Index.restore('db5.usearch', view=True) 

MODEL_NAME = "all-mpnet-base-v2"  # Fast & lightweight; swap for a larger model if needed
encoder = SentenceTransformer(MODEL_NAME)

while True:
    prompt = input("You: ")
    
    if prompt.lower() in {"exit", "quit"}:
        break

    matches = index.search(encoder.encode(prompt), 3)
    context = [sentences[m.key] for m in matches]

    # Append user message
    prompt = f"""
"<start_of_turn>user
Input: {prompt}

Conversation History:
{"\n".join(messages[-3:])}

World Context:
{"\n".join(context)}<end_of_turn>"
"""
    print(prompt.replace("\n", " "))
