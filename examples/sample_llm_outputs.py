import argparse
import glob
import os

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--llm",
    type=str,
    default="gpt2",
    choices=["gpt2", "llama2", "llama3", "gemma", "mistral"],
)
parser.add_argument(
    "--dataset", type=str, default="wiki", choices=["cnn", "wiki", "reddit"]
)
parser.add_argument("--n_samples", type=int, default=100)
args = parser.parse_args()

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
alphabet_size = tokenizer.vocab_size

## Dataset
if args.dataset == "cnn":
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train")
    sent_key = "article"
    max_len = 300
    n_tokens = 60
elif args.dataset == "reddit":
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
    sent_key = "prompt"
    max_len = -1
    n_tokens = 40
else:
    dataset = load_dataset("tdtunlp/wikipedia_en")["train"]
    sent_key = "text"
    max_len = 200
    n_tokens = 20

dataset = dataset.shuffle(seed=42)

RES_DIR = f".cache/llm_output_samples/{args.dataset}_{args.llm}"
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

for i, sentence in tqdm.tqdm(enumerate(dataset)):
    idx = str(i)
    model_input = tokenizer(sentence[sent_key], return_tensors="pt")["input_ids"].to(
        DEVICE
    )
    model_input = model_input[:, :max_len]

    for d in range(n_tokens):
        with torch.no_grad():
            model_output = torch.softmax(model(input_ids=model_input).logits, dim=-1)

        sample = model_output[0, -1, :]
        index = torch.argmax(sample)
        model_input = torch.cat([model_input, index.expand(1, 1)], dim=1)

        torch.save(sample.cpu(), f"{RES_DIR}/sample_index{idx}_depth{d}.pt")

# Stack them together into a (n_samples*n_tokens, vocab_size) tensor
sample_files = glob.glob(f"{RES_DIR}/sample_*.pt")
samples = [torch.load(sample) for sample in sample_files]
torch.save(torch.vstack(samples), f"{RES_DIR}/all_samples.pt")
