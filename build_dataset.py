from usearch.index import Index
from sentence_transformers import SentenceTransformer
import random, json

def sliding_windows(arr, max_window = 4):
    arr = [x for x in arr if x != ""]

    for i in range(1, min(max_window, len(arr)) + 1):
        yield arr[:i]

    for i in range(1, len(arr) - max_window + 1):
        yield arr[i:i+max_window]

random.seed(1234)

conversations = "pairs"
fixed_conversations = "fixed_conversations"
worlds = ["db1", "db2", "db3", "db4", "db5"]

conversations_index = Index.restore(f'{conversations}.usearch', view=False) 
with open(f"{conversations}.txt", "r") as f:
    conversations = f.read().splitlines()
with open(f"{fixed_conversations}.txt", "r") as f:
    fixed_conversations = [x.splitlines() for x in f.read().split("~")]
world_indices = [Index.restore(f'{x}.usearch', view=False) for x in worlds]
world_sentences = []
for w in worlds:
    with open(f"{w}.txt", "r") as f:
        world_sentences.append(f.read().splitlines())

with open(f"prompt_template.txt", "r") as f:
    prompt_template = f.read()

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Build from pairs
prompts = []
for i, c in enumerate(conversations):
    (input, response) = c.split("->")
    vector = model.encode([input], show_progress_bar=False)

    matches = conversations_index.search(vector, 9)
    keys = []
    for match in matches:
        key = int(match.key / 2)
        if key == i: continue
        if input in conversations[key]: continue
        if key not in keys:
            keys.append(key)

    for index, sentences in zip(world_indices, world_sentences):
        history = "\n".join([conversations[key] for key in random.sample(keys, 4)])

        matches = index.search(vector, 3)
        ctx_keys = [m.key for m in matches]
        context = "\n".join([sentences[key] for key in ctx_keys])

        prompt = (prompt_template.replace("{input}", input)
            .replace("{history}", history)
            .replace("{context}", context)
            .replace("{response}", response))
        prompts.append(prompt)

# Build from fixed conversations
for conversation in fixed_conversations:
    for c in sliding_windows(conversation):
        input = c[-1]
        input, response = [x.strip("\n\t\" ") for x in input.split("->")]
        if len(c) > 1:
            history_items = c[0:-1]
        else: history_items = []
        history_items = [x.split("->")[0].strip("\n\t\" ") + " -> " + x.split("->")[1].strip("\n\t\" ") for x in history_items]

        while len(history_items) < 4:
            history_items.append("<none>")
        
        for index, sentences in zip(world_indices, world_sentences):

            vector = model.encode([input], show_progress_bar=False)

            matches = index.search(vector, 3)
            ctx_keys = [m.key for m in matches]
            context = "\n".join([sentences[key] for key in ctx_keys])

            prompt = (prompt_template.replace("{input}", input)
                .replace("{history}", history)
                .replace("{context}", context)
                .replace("{response}", response))
            prompts.append(prompt)

print(json.dumps(prompts))


