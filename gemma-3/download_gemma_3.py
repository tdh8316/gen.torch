import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_LIST = {
    "gemma-3-270m-it": "https://huggingface.co/google/gemma-3-270m-it",
    "gemma-3-1.3b-it": "https://huggingface.co/google/gemma-3-1.3b-it",
}

print("Available models:")
for index, (key, _) in enumerate(MODEL_LIST.items()):
    print(f"{index + 1}. {key}")
model_choice = input("Select a model by number: ")
try:
    model_index = int(model_choice) - 1
    MODEL_PATH = list(MODEL_LIST.values())[model_index]
except (ValueError, IndexError):
    print("Invalid selection. Exiting.")
    exit(1)

print(f"[i] Model path: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=f"./.cache")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir=f"./.cache")

model_name = MODEL_PATH.split("/")[-1]
print(f"[i] Loaded {model_name}")

messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print("=" * 40)
print(
    ("=" * 40 + "\n"),
    ">>>",
    tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ),
)

# Save raw PyTorch model
torch.save(model.state_dict(), f"./{model_name}.pt")
print(f"[i] Saved the model to ./{model_name}.pt")
