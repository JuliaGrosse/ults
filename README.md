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

See full example here: [examples/generate.py](https://github.com/JuliaGrosse/ults/blob/main/examples/generate.py).

```python
from ults import ULTS

tokenizer = AutoTokenizer.from_pretrained("meta/Llama-2-7b-hf").to(DEVICE)
model = AutoModelForCausalLM.from_pretrained("meta/Llama-2-7b-hf", torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

text = "Moose is a"
model_inputs = tokenizer(text, return_tensors="pt")

ults = ULTS(
    model=model,
    model_inputs=model_inputs,
    max_tokens=20,
    vocab_size=tokenizer.vocab_size,
    device="cuda"
)
best_sequence, best_loglik, n_llm_calls = ults.search()

context_len = model_inputs["input_ids"].shape[-1]
generated_tokens = best_sequence[0, -(n + 1) :]
generated_text = tokenizer.decode(generated_tokens)
```

### Empirical Prior

TODO!

## Caveats

1. Currently doesn't support batch generation.
2. Currently only supports Huggingface's `AutoModelForCausalLM`.
   1. E.g., supports Llama-2, Llama-3, Gemma, GPTs, Mistrals, etc.
   2. But, encoder-decoder architectures like T5 are not currently supported.

## License

```text
Copyright (c) 2024 Julia Grosse and Agustinus Kristiadi

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
