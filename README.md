# TinyLlama QLoRA Fine-Tuning

This project demonstrates parameter-efficient fine-tuning of TinyLlama using QLoRA and Hugging Face PEFT.

## Features

- TinyLlama 1.1B
- 4-bit Quantization
- LoRA Adapters
- Hugging Face Hub Integration
- Inference Script

## Tech Stack

- Python
- Transformers
- PEFT
- BitsAndBytes
- Hugging Face Hub

## Training Stats

Total Parameters: 1.1B

Trainable Parameters: 2.25M

Trainable %: 0.20%

## Adapter Repository

https://huggingface.co/TejaswiDongala/tinyllama-qlora <img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/73381433-f623-42a2-9ab8-dead200e9779" />

## requirements.txt ---
torch

transformers

datasets

accelerate

peft

trl

bitsandbytes

huggingface_hub

sentencepiece

protobuf

safetensors
gradio
