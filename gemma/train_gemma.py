from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    # model_name = "unsloth/gemma-3-4b-it",
    model_name = "unsloth/gemma-3-12b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma-3",
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
    return {"text": "<bos>" + entry + "<eos>"}

parsed = [parse_entry(e) for e in raw]
dataset = Dataset.from_list(parsed)

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

# def formatting_prompts_func(examples):
#    convos = examples["conversations"]
#    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
#    return { "text" : texts, }

# dataset = dataset.map(formatting_prompts_func, batched = True)

print(dataset[100])

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "Input:",
    response_part = "Response:",
)

print("training")
trainer_stats = trainer.train()

model.save_pretrained("holo_gemma3")  # Local saving
tokenizer.save_pretrained("holo_gemma3")

model.save_pretrained_gguf(
    "holo_gemma3",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)