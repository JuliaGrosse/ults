"""Utilities."""

def n_grams(tokens, n) -> int:
    """n_grams in the token sequence."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def rep_n(tokens, n) -> float:
    """portion of duplicate n-grams. (Also see: https://arxiv.org/pdf/2202.06417)"""
    total_ngrams = n_grams(tokens, n)
    unique_ngrams = set(total_ngrams)
    return 100 * (1 - len(unique_ngrams) / len(total_ngrams))
