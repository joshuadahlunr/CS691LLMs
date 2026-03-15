from unsloth import FastModel
import torch

# =========================
# LOAD BASE MODEL
# =========================
model, tokenizer = FastModel.from_pretrained(
    # model_name = "unsloth/gemma-3-4b-it",
    model_name = "unsloth/gemma-3-12b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

# =========================
# LOAD LORA ADAPTER
# =========================
model = FastModel.get_peft_model(
    model,
    r=16,  # ignored when loading pretrained LoRA, but required
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# Load trained weights
model.load_adapter("./holo_gemma3", adapter_name="holo")
model.set_adapter("holo")
print(model.active_adapters)

# Set to inference mode
FastModel.for_inference(model)

# =========================
# RUN INFERENCE
# =========================
prompt = """
Input: Why is the sky blue?

Conversation History:
<none>
<none>
<none>

World Context:
She is knowledgeable about the natural world.
She has long brown hair and striking reddish eyes.
Holo leaves her village in search of her northern homeland, Yoitsu.

"""



messages = ["<none>"] * 3

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

from usearch.index import Index
from sentence_transformers import SentenceTransformer

with open("../db5.txt", "r") as f:
    sentences = f.read().splitlines()

index = Index.restore('../db5.usearch') 

MODEL_NAME = "all-mpnet-base-v2"  # Fast & lightweight; swap for a larger model if needed
encoder = SentenceTransformer(MODEL_NAME)

embeddings = encoder.encode(messages)
convo_index = Index(ndim=embeddings.shape[1])
for i in range(0, 3):
    convo_index.add(i, embeddings[i])

while True:
    user_input = input("You: ")
    
    if user_input.lower() in {"exit", "quit"}:
        break

    embedding = encoder.encode(user_input)
    matches = index.search(embedding, 3)
    context = [sentences[m.key] for m in matches]

    matches = convo_index.search(embedding, 2)
    history = [messages[m.key] for m in matches]
    history = list(set(history + messages[-2:]))
    while len(history) < 4:
        history = ["<none>"] + history

    # Append user message
    prompt = f"""
Input: {user_input}

Conversation History:
{"\n".join(history)}

World Context:
{"\n".join(context)}

Response:
"""
    print([prompt])

    # Tokenize with chat template
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_dict=True,
    )

    # Generate response
    output = model.generate(
        **inputs.to("cuda"),
        max_new_tokens=64,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=streamer,
    )

    # Decode full output to extract assistant reply
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the assistant's latest response
    response = decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    full = f"{user_input} -> {response}"
    convo_index.add(len(messages), encoder.encode([full]))
    messages.append(full)
