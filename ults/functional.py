from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass

import torch

from ults.ults import ULTS


@dataclass
class ULTSOutput:
    sequence: torch.Tensor
    loglik: float
    n_llm_calls: int


def generate(
    model: torch.nn.Module,
    model_inputs: UserDict,
    max_tokens: int,
    max_beam_size: int = 5,
    vocab_size: int | None = None,
    epsilon: float = 0.1,
    prior_kind: str = "dirichlet",
    prior_dirichlet_alpha: float = 0.0001,
    prior_empirical_dataset_name: str | None = None,
    prior_empirical_llm_samples: torch.Tensor | None = None,
    prior_dir: str = "./ults_priors",
    sample_size: int = 1000,
    output_full_sequence: bool = False,
    stop_at_eos: bool = True,
    acquisition_function: str = "posterior",
) -> ULTSOutput:
    """ULTS: Uncertainty-guided Likelihood-Tree Search.

    Args:
        model: A Huggingface LLM model.
        model_inputs: The input of `model(...)` or `model.forward(...)`.
            Must contain key "input_ids".
        max_tokens: Maximum number of tokens to generate.
        max_beam_size: Maximum number of nodes to expand per level.
        vocab_size: Vocabulary size. This should be `len(tokenizer)` in most cases.
            If `None`, then this will be inferred from `model.config.vocab_size`.
        epsilon: Confidence level for termination.
        prior_kind: "dirichlet" or "empirical".
        prior_dir: The location of the cached priors.
        prior_dirichlet_alpha: Concentration parameter of the Dirichlet prior.
        prior_empirical_dataset_name: Dataset name where `prior_empirical_llm_samples`
            are obtained from.
        prior_empirical_llm_samples: LLM output samples for the empirical prior.
        sample_size: Number of posterior samples to use.
        output_full_sequence: Whether to output the full sequence (context + generated).
            The outputted loglik will reflect this.
        stop_at_eos: Consider sequences that end with <EOS> as leaf nodes.
        acquisition_function: "posterior" or "posterior_descendant".
            "posterior": pick child node based on posterior over max loglik.
            "posterior_descendant": pick child node based on posterior over max loglik
            of the best descendant.

    Returns:
        ults_output: A dataclass containing `sequence`, `loglik`, and `n_llm_calls`.
    """
    ults = ULTS(
        model=model,
        model_inputs=model_inputs,
        max_tokens=max_tokens,
        max_beam_size=max_beam_size,
        vocab_size=vocab_size,
        epsilon=epsilon,
        prior_kind=prior_kind,
        prior_dir=prior_dir,
        prior_dirichlet_alpha=prior_dirichlet_alpha,
        prior_empirical_dataset_name=prior_empirical_dataset_name,
        prior_empirical_llm_samples=prior_empirical_llm_samples,
        sample_size=sample_size,
        stop_at_eos=stop_at_eos,
        acquisition_function=acquisition_function,
    )

    # Generation results --- full sequence and total_loglik include context
    full_sequence, total_loglik, n_llm_calls = ults.search()

    if output_full_sequence:
        sequence = full_sequence
        loglik = total_loglik
    else:
        sequence = full_sequence[:, len(ults.tree.nodes["0"]["tokens"][0, :]) :]
        loglik = total_loglik - ults.tree.nodes["0"]["loglike"]

    return ULTSOutput(
        sequence=sequence,
        loglik=loglik,
        n_llm_calls=n_llm_calls,
    )
