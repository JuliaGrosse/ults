# ULTS: Uncertainty-guided Likelihood Tree Search

Accompanying implementation of the following paper [[ArXiv]](TODO!):

```bib
@article{grosse2024ults,
  title={Uncertainty-Guided Optimization On Large Language Model Search Trees},
  author={Grosse, Julia and Wu, Ruotian and Rashid, Ahmad and Hennig, Philipp and Poupart, Pascal and Kristiadi, Agustinus},
  journal={arXiv preprint arXiv:TODO!},
  year={2024}
}
```

## Setup

Requires Python > 3.10.

1. Install PyTorch with CUDA, version >= 2.0
2. Install this repo: `pip install git+https://github.com/JuliaGrosse/ults.git@main`

## Usage

See full example here: [examples/generate.py](https://github.com/JuliaGrosse/ults/blob/main/examples/generate.py). Run via `python examples/generate.py`. You need to adapt the `from_pretrained` lines to point Huggingface or your own local cache.

#### Quickstart snippet

```diff
from ults import ULTS

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf").to(DEVICE)
model = AutoModelForCausalLM.from_pretrained(
  "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
).to(DEVICE)
model.eval()

text = "Moose is a"
model_inputs = tokenizer(text, return_tensors="pt")

-# Beam search
-output = model.generate(
-    **model_inputs,
-    num_beams=5,
-    max_new_tokens=40,
-)

+# ULTS
+ults = ULTS(
+    model=model,
+    model_inputs=model_inputs,
+    max_tokens=40,
+    vocab_size=tokenizer.vocab_size,
+)
+best_sequence, best_loglik, n_llm_calls = ults.search()

context_len = model_inputs["input_ids"].shape[-1]
generated_tokens = best_sequence[0, context_len:]
generated_text = tokenizer.decode(generated_tokens)
```

### Using the Empirical Prior

TODO!

## Caveats

1. Currently doesn't support batch generation.
2. Currently only supports Huggingface's `AutoModelForCausalLM`.
   1. E.g., supports Llama-2, Llama-3, Gemma, GPTs, Mistrals, etc.
   2. But, encoder-decoder architectures like T5 are not currently supported.
