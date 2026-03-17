from unsloth import FastModel
import torch

# =========================
# LOAD BASE MODEL
# =========================
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

# =========================
# LOAD LORA ADAPTER
# =========================
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Load trained weights
model.load_adapter("./holo_llama3.2", adapter_name="holo")
model.set_adapter("holo")
print(model.active_adapters)

# =========================
# MERGE ADAPTER INTO MODEL
# =========================
print("Merging adapter weights into base model...")

# Merge and unload returns a standard HuggingFace model with weights merged
merged_model = model.merge_and_unload()

print("Merge complete.")

# =========================
# SAVE MERGED MODEL
# =========================
save_path = "./holo_llama3.2_merged"

print(f"Saving merged model to {save_path} ...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

merged_model.save_pretrained_gguf(
    "holo_llama3.2",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)

print(f"Done! Merged model saved to: {save_path}")