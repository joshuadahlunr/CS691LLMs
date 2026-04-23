from unsloth import FastModel
import torch
import random

# Set torch to start at a specific seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
random.seed(123)

model, tokenizer = FastModel.from_pretrained(
    # model_name = "unsloth/gemma-3-4b-it",
    model_name = "unsloth/Phi-4",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

model = FastModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "qwen3.5",
# )

from datasets import Dataset
# from datasets import load_dataset
# dataset = load_dataset("json", data_files="dataset.txt", split="train")
import json, random
random.seed(1234)
with open("../dataset.txt", "r") as f:
    raw = json.load(f)
raw = random.sample(raw, len(raw))

def parse_entry(entry):
    # Extract the response after "Response: "
    response = entry.split("Response:")[-1].strip()
    # Extract the input/context as the user turn
    user_turn = entry.split("Response:")[0].strip()
    # return {"conversations": [
    #     {"role": "user", "content": user_turn},
    #     {"role": "assistant", "content": response}
    # ]}
    return {"text": entry + "<|im_end|>"}

parsed = [parse_entry(e) for e in raw]
dataset = Dataset.from_list(parsed)

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

# def format_chat(examples):
#     texts = tokenizer.apply_chat_template(
#         examples["messages"],
#         tokenize=False,
#         add_generation_prompt=False
#     )
#     return {"text": texts}

# dataset = dataset.map(format_chat)

print(dataset[100])

from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "Input:",
    response_part = "Response:",
)

print("training...")
trainer_stats = trainer.train()

model.save_pretrained("holo_phi4")  # Local saving
tokenizer.save_pretrained("holo_phi4")

model.save_pretrained_gguf(
    "holo_phi4",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)