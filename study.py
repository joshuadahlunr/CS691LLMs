#!/usr/bin/env python3
"""
Multi-LLM Chat with Ollama
- Chats with 4 different LLMs in a permuted order based on user ID
- Type /next to switch to the next LLM
- Chat log is saved automatically on exit or after the last model
"""

import json
import hashlib
import itertools
import ollama
from datetime import datetime
from pathlib import Path
from usearch.index import Index
from sentence_transformers import SentenceTransformer


# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = [
    "llama3.2",
    "mistral",
    "gemma3",
    "qwen2.5",
]

NEXT_COMMAND = "/next"
QUIT_COMMAND = "/quit"

# ── Load RAG ─────────────────────────────────────────────────────────────

with open("db6.txt", "r") as f:
    sentences = f.read().splitlines()

index = Index.restore('db6.usearch', view=False) 

MODEL_NAME = "all-mpnet-base-v2"
encoder = SentenceTransformer(MODEL_NAME)

embeddings = encoder.encode(["hello"])
convo_index = Index(ndim=embeddings.shape[1])

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_model_order(user_id: int, models: list[str]) -> list[str]:
    """
    Deterministically permute the model list based on user_id.
    Uses a hash of the user_id as a seed index into all permutations.
    """
    all_perms = list(itertools.permutations(models))
    chosen = list(all_perms[user_id % len(all_perms)])
    return chosen


def check_model_available(model: str) -> bool:
    """Return True if the model is pulled and available locally."""
    try:
        local_models = [m.model for m in ollama.list().models]
        # Ollama tags can include ':latest', so check prefix match too
        return any(m == model or m.startswith(model + ":") for m in local_models)
    except Exception:
        return False


def chat_with_model(model: str, prompt: str, messages: list[str]) -> str:
    """
    Send the full conversation history to the model and return
    (assistant_reply, updated_history).
    """
    embedding = encoder.encode(prompt)
    matches = index.search(embedding, 3)
    context = [sentences[m.key] for m in matches]

    matches = convo_index.search(embedding, 2)
    history = [messages[m.key] for m in matches]
    history = list(set(history + messages[-2:]))
    while len(history) < 4:
        history = ["<none>"] + history

    message = [{
        "role": "user",
        "content": f"""
Input: {prompt}

Conversation History:
{"\n".join(history)}

World Context:
{"\n".join(context)}
"""
    }]
    # print(message)
    response = ollama.chat(model=model, messages=messages)
    reply = response["message"]["content"]
    full = f"{prompt} -> {reply}"
    convo_index.add(len(messages), encoder.encode([full]))
    return reply


def save_log(user_id: int, model_order: list[str], sessions: list[dict]) -> Path:
    """Save the full chat log as a JSON file and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(f"chat_log_{user_id}_{timestamp}.json")

    log = {
        "user_id": user_id,
        "timestamp": timestamp,
        "model_order": model_order,
        "sessions": sessions,
    }

    filename.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    return filename


def print_banner(user_id: int, model_order: list[str]) -> None:
    print("\n" + "═" * 60)
    print("  Multi-LLM Chat  │  Powered by Ollama")
    print("═" * 60)
    print(f"  User ID   : {user_id}")
    # print(f"  LLM order : {' → '.join(model_order)}")
    print("─" * 60)
    print(f"  Commands  : {NEXT_COMMAND} = switch model  │  {QUIT_COMMAND} = exit & save")
    print("═" * 60 + "\n")


def print_model_header(index: int, total: int, model: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Model {index}/{total}: {model}")
    print(f"{'─'*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Get user ID
    print("\n╔══════════════════════════════╗")
    print("║    Multi-LLM Chat (Ollama)   ║")
    print("╚══════════════════════════════╝\n")
    user_id = input("Enter your user ID: ").strip()
    if not user_id:
        print("User ID cannot be empty.")
        return
    user_id = int(user_id)

    # 2. Determine model order
    model_order = get_model_order(user_id, MODELS)

    print_banner(user_id, model_order)

    # 3. Check which models are available
    unavailable = [m for m in model_order if not check_model_available(m)]
    if unavailable:
        print("⚠  The following models are not pulled locally:")
        for m in unavailable:
            print(f"   • {m}  →  run: ollama pull {m}")
        print()
        proceed = input("Continue anyway? Unavailable models will be skipped. [y/N]: ").strip().lower()
        if proceed != "y":
            print("Exiting.")
            return

    # 4. Chat loop
    sessions: list[dict] = []   # one entry per model

    for idx, model in enumerate(model_order, start=1):
        if not check_model_available(model):
            print(f"\n⚠  Skipping {model} (not available locally).")
            sessions.append({"model": model, "skipped": True, "messages": []})
            continue

        # print_model_header(idx, len(model_order), model)
        # print(f"You are now chatting with {model}.")
        print(f"Type {NEXT_COMMAND} to move to the next model, or {QUIT_COMMAND} to exit.\n")

        history: list[str] = []
        session_done = False
        quit_early = False

        while not session_done:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n(Interrupted)")
                quit_early = True
                session_done = True
                break

            if not user_input:
                continue

            if user_input.lower() == NEXT_COMMAND:
                if idx < len(model_order):
                    print(f"\n→ Moving to the next model…\n")
                else:
                    print(f"\n✓ That was the last model.\n")
                session_done = True
                break

            if user_input.lower() == QUIT_COMMAND:
                quit_early = True
                session_done = True
                break

            # Query model
            try:
                print(f"\n{model}: ", end="", flush=True)
                reply = chat_with_model(model, user_input, history)
                history.append(f"{user_input} -> {reply}")
                print(reply)
                print()
            except Exception as e:
                print(f"\n⚠  Error querying {model}: {e}\n")
                # Remove the last user message so the conversation stays consistent
                history = history[:-1]
                continue

        sessions.append({"model": model, "skipped": False, "messages": history})

        if quit_early:
            break

    # 5. Save log
    log_path = save_log(user_id, model_order, sessions)
    print("\n" + "═" * 60)
    print(f"  Chat log saved → {log_path}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()