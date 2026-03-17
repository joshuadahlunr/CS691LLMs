from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-14B-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastLanguageModel.get_peft_model(
    model,
    # finetune_vision_layers     = False, # False if not finetuning vision layers
    # finetune_language_layers   = True, # False if not finetuning language layers
    # finetune_attention_modules = True, # False if not finetuning attention layers
    # finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)
# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "phi-4",
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
    return {"text": "<|im_start|>" + entry + "<|im_end|>"}

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
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    # max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    aargs = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # output_dir = "outputs",
        # report_to = "none", # Use this for WandB etc
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

model.save_pretrained("holo_qwen3.5")  # Local saving
tokenizer.save_pretrained("holo_qwen3.5")

model.save_pretrained_gguf(
    "holo_qwen3.5",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)