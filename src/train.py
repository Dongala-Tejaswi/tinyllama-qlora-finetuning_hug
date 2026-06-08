


import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

with open("../data/train.json","r") as f:
    data = json.load(f)

texts = []

for item in data:
    texts.append(
        f"Instruction: {item['instruction']}\nResponse: {item['response']}"
    )

dataset = Dataset.from_dict({"text":texts})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
model.save_pretrained("../adapter")
tokenizer.save_pretrained("../adapter")

model.push_to_hub(
    "TejaswiDongala/tinyllama-qlora"
)
tokenizer.push_to_hub(
    "TejaswiDongala/tinyllama-qlora"
)