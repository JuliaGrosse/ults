import os
import argparse
import torch
import numpy as np
import glob
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from ults import ULTS
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--llm",
    type=str,
    default="llama2",
    choices=["gpt2", "llama2", "llama3", "gemma", "mistral"],
)
args = parser.parse_args()

np.random.seed(19)
torch.manual_seed(19)
random.seed(19)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.llm == "llama2":
    MODEL_NAME = "Llama-2-7b-hf"
elif args.llm == "llama3":
    MODEL_NAME = "Meta-Llama-3-8B"
elif args.llm == "gemma":
    MODEL_NAME = "gemma-7b"
elif args.llm == "mistral":
    MODEL_NAME = "Mistral-7B-v0.1"
else:
    MODEL_NAME = "gpt2"

# TODO: Make sure you update `MODEL_PATH` and `local_files_only=True` below!
MODEL_PATH = f"/model-weights/{MODEL_NAME}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
).to(DEVICE)

model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

context = "Predominantly a browser, the moose's diet consists of"
model_inputs = tokenizer(context, return_tensors="pt")

ults = ULTS(
    model=model,
    model_inputs=model_inputs,
    max_tokens=40,
    vocab_size=tokenizer.vocab_size,
    max_beam_size=5,
    epsilon=0.1,
    prior_kind="dirichlet",
    prior_dirichlet_alpha=0.0001,
    sample_size=1000,
)
# Generation results
sequence, total_loglik, n_llm_calls = ults.search()

# Evaluate log likelihood of generated tokens
context_len = model_inputs["input_ids"].shape[-1]
generated_tokens = sequence[0, context_len:]

with torch.no_grad():
    logprobs = torch.log_softmax(model(sequence).logits, dim=-1)

# Logprobs of the generated tokens only (without the context)
# The `-1` here is because the last seq. index of the logits is for the next word
# *after* the last generated word (something we didn't generate!)
logprobs = logprobs[0, (context_len - 1) : -1, :]
loglik = torch.sum(logprobs[torch.arange(len(logprobs)), generated_tokens]).item()

# Print results
print()
print(f'Context: "{context}"')
print(
    f"Loglik: {loglik:.4f}, total loglik: {total_loglik:.4f}, num. LLM calls: {n_llm_calls}"
)
print(f'Generated: "{tokenizer.decode(generated_tokens)}"')
print()
