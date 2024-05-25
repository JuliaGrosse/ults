import os
import argparse
import torch
import numpy as np
import glob
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import tqdm
from ults import ULTS
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prior", type=str, default="dirichlet", choices=["dirichlet", "empirical"]
)
parser.add_argument("--epsilon", type=float, default=0.1, help="confidence parameter")
parser.add_argument(
    "--alpha", type=float, default=0.0001, help="concentration parameter"
)
parser.add_argument("--maxbeam", type=int, default=5, help="maximum beam size")
parser.add_argument(
    "--llm",
    type=str,
    default="llama2",
    choices=["gpt2", "llama2", "llama3", "gemma", "mistral"],
)
parser.add_argument("--ntokens", type=int, help="number of tokens to append")
parser.add_argument(
    "--dataset", type=str, default="wiki", choices=["cnn", "wiki", "reddit"]
)
parser.add_argument("--sample_size", type=int, default=1000)
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

MODEL_PATH = f"/model-weights/{MODEL_NAME}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
).to(DEVICE)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
ALPHABET_SIZE = tokenizer.vocab_size

########################################################################################
# Generate dataset for empirical prior
########################################################################################
if args.prior == "empirical":
    stacked_samples_file = (
        f"llm_output_samples/{args.dataset}_{args.llm}/stacked_samples.pt"
    )
    if os.path.exists(stacked_samples_file):
        samplesllm = torch.load(stacked_samples_file, map_location=torch.device(DEVICE))
    else:
        print("Stacking LLM output samples...")
        llsamples_files = glob.glob(
            f"llm_output_samples/{args.dataset}_{args.llm}/qualities_*.pt"
        )
        random.shuffle(llsamples_files)
        llsamples_files = llsamples_files[: args.sample_size]
        collected_qualities = []

        for qualities in tqdm.tqdm(llsamples_files):
            qualities = torch.load(qualities, map_location=torch.device(DEVICE))
            collected_qualities.append(qualities)

        samplesllm = torch.vstack(collected_qualities)
        torch.save(samplesllm, stacked_samples_file)
########################################################################################

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
    device="cuda",
)

sequence, _, n_llm_calls = ults.search()

########################################################################################
# Evaluate log likelihood of generated tokens
########################################################################################
context_len = model_inputs["input_ids"].shape[-1]
generated_tokens = sequence[0, context_len:]

with torch.no_grad():
    logprobs = torch.log_softmax(model(sequence).logits, dim=-1)

# Logprobs of the generated tokens only (without the context)
logprobs = logprobs[0, context_len:, :]
loglik = torch.sum(logprobs[torch.arange(len(logprobs)), generated_tokens]).item()

print(f'Context: "{context}"')
print(f"Loglik: {loglik:.4f}")
print(f'Generated: "{tokenizer.decode(generated_tokens)}"')
