import argparse
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

import ults

parser = argparse.ArgumentParser()
parser.add_argument(
    "--llm",
    type=str,
    default="llama2",
    choices=["gpt2", "llama2", "llama3", "gemma", "mistral", "t5"],
)
parser.add_argument("--weight-path", type=str, default=None)
args = parser.parse_args()

np.random.seed(19)
torch.manual_seed(19)
random.seed(19)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.llm == "llama2":
    HF_PATH = "meta-llama"
    MODEL_NAME = "Llama-2-7b-hf"
elif args.llm == "llama3":
    HF_PATH = "meta-llama"
    MODEL_NAME = "Meta-Llama-3-8B"
elif args.llm == "gemma":
    HF_PATH = "google"
    MODEL_NAME = "gemma-7b"
elif args.llm == "mistral":
    HF_PATH = "mistralai"
    MODEL_NAME = "Mistral-7B-v0.1"
elif args.llm == "t5":
    HF_PATH = "google-t5"
    MODEL_NAME = "t5-base"
else:
    HF_PATH = "openai-community"
    MODEL_NAME = "gpt2"

LOCAL_FILE = args.weight_path is not None
MODEL_PATH = f"{args.weight_path if LOCAL_FILE else HF_PATH}/{MODEL_NAME}"
MODEL_CLS = AutoModelForSeq2SeqLM if args.llm == "t5" else AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=LOCAL_FILE, model_max_length=512
)
model = MODEL_CLS.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=LOCAL_FILE
).to(DEVICE)

model.eval()

if args.llm != "t5":
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

if args.llm == "t5":
    context = "Translate English to German: My name is Wolfgang and I live in Berlin"
else:
    context = "Predominantly a browser, the moose's diet consists of"

model_inputs = tokenizer(context, return_tensors="pt")

output = ults.generate(
    model=model,
    model_inputs=model_inputs,
    max_tokens=40,
    max_beam_size=5,
    epsilon=0.1,
    prior_kind="dirichlet",
    prior_dirichlet_alpha=0.0001,
    prior_dir="./ults_priors",
    sample_size=1000,
    output_full_sequence=False,
    acquisition_function="posterior",
)

# Print results
print()
print(f'Context: "{context}"')
print(f"Loglik: {output.loglik:.4f}, num. LLM calls: {output.n_llm_calls}")
print(f'Generated: "{tokenizer.decode(output.sequence[0], skip_special_tokens=True)}"')
print()
