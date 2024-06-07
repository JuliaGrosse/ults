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

Requires Python >= 3.10.

1. Install PyTorch with CUDA, version >= 2.0
2. Install Huggingface libraries: `pip install transformers datasets tokenizers`
3. Install this repo: `pip install git+https://github.com/JuliaGrosse/ults.git@main`

## Usage

See full example here: [examples/generate.py](https://github.com/JuliaGrosse/ults/blob/main/examples/generate.py).

### Quickstart with the Dirichlet prior

```diff
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
  "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
)
model.eval()

text = "Moose is a"
model_inputs = tokenizer(text, return_tensors="pt")

-output = model.generate(
-    **model_inputs,
-    num_beams=5,
-    max_new_tokens=40,
-)
-generated_sequence = output.sequences

+import ults
+output = ults.generate(
+    model=model,
+    model_inputs=model_inputs,
+    max_tokens=40,
+    vocab_size=tokenizer.vocab_size,
+)
+generated_sequence = output.sequence

generated_text = tokenizer.decode(generated_sequence[0])
```

### Using the Empirical Prior

On top of the default Dirichlet priors (agnostic to the LLM), ULTS can also leverage
empirical priors, specific to the LLM at hand.
Example precomputed empirical priors, compatible with [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), and [Gemma-7b](https://huggingface.co/google/gemma-7b), are available in `examples/.cache/priors`.

1. First, gather samples of the LLM's softmax outputs from different time steps. Here
   we will use the greedy decoding. See `examples/sample_llm_outputs.py` for a complete example

```python
RESULT_DIR = f".cache/llm_output_samples/{DATASET_NAME}_{LLM_NAME}"

# Samples of contexts from your dataset
contexts: List[str]

for idx, context in enumerate(contexts):
    input_ids = tokenizer(sentence[sent_key], return_tensors="pt")["input_ids"]

    # `n_tokens` is the max. depth of the tree that you want to optimize on
    # i.e., the max number of tokens you want to generate with ULTS
    for d in range(n_tokens):
        with torch.no_grad():
            outputs = torch.softmax(model(input_ids).logits, dim=-1)

        # Save the last softmax output (this is our empirical sample for depth `d`)
        outputs = outputs[0, -1, :]
        torch.save(outputs, f"{RESULT_DIR}/sample_index{idx}_depth{d}.pt")

        # Continue greedy generation
        index = torch.argmax(qualities)
        model_input = torch.cat([model_input, index.expand(1, 1)], dim=1)

# Stack them together into a (n_samples*n_tokens, vocab_size) tensor
import glob, random
sample_files = glob.glob(f"{RESULT_DIR}/sample_*.pt")
samples = [torch.load(sample) for sample in sample_files]
torch.save(torch.vstack(samples), f'{RESULT_DIR}/all_samples.pt')
```

2. Then, when specify the prior when calling ULTS. Everything else stays the same as in
   `examples/generate.py`.

```diff
output = ults.generate(
    ...
+   prior_kind="empirical",
+   prior_empirical_llm_samples=torch.load(f'{RESULT_DIR}/all_samples.pt')
    ...
)
```

## Caveats

1. Currently doesn't support batch generation.
