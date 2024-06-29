from __future__ import annotations

import os

import networkx as nx
import numpy as np
import torch
import tqdm
import math
from scipy import stats
from torch.distributions.beta import Beta
from transformers import BatchEncoding


class ULTS:
    """ULTS: Uncertainty-guided Likelihood-Tree Search.

    Args:
        model: A Huggingface LLM model.
        model_inputs: The input of `model(...)` or `model.forward(...)`.
            Must contain key "input_ids". This is usually the output of `tokenizer(text)`.
        max_tokens: Maximum number of tokens to generate.
        vocab_size: Vocabulary size. This should be `len(tokenizer)` in most cases.
            If `None`, then this will be inferred from `model.config.vocab_size`.
        max_beam_size: Maximum number of nodes to expand per level.
        epsilon: Confidence level for stopping criterion.
        prior_kind: "dirichlet" or "empirical".
        prior_dir: The location of the cached priors.
        prior_dirichlet_alpha: Concentration parameter of the Dirichlet prior.
        prior_empirical_dataset_name: Dataset name where `prior_empirical_llm_samples`
            are obtained from.
        prior_empirical_llm_samples: LLM output samples for the empirical prior.
        sample_size: Number of posterior samples to use.
        stop_at_eos: Consider sequences that end with <EOS> as leaf nodes.
        acquisition_function: "posterior" or "posterior_descendant".
            "posterior": pick child node based on posterior over max loglik.
            "posterior_descendant": pick child node based on posterior over mx loglik
            of best descendant.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_inputs: BatchEncoding,
        max_tokens: int,
        max_beam_size: int = 5,
        vocab_size: int | None = None,
        epsilon: float = 0.1,
        prior_kind: str = "dirichlet",
        prior_dir: str = "./ults_priors",
        prior_dirichlet_alpha: float = 0.0001,
        prior_empirical_dataset_name: str | None = None,
        prior_empirical_llm_samples: torch.Tensor | None = None,
        sample_size: int = 1000,
        stop_at_eos: bool = True,
        acquisition_function: str = "posterior",
    ):
        if prior_kind == "empirical" and prior_empirical_dataset_name is None:
            raise ValueError(
                "`prior_empirical_dataset_name` cannot be `None` for empirical prior."
            )

        if acquisition_function not in ["posterior", "posterior_descendant"]:
            raise ValueError(
                "`acquisition_function` can only be `posterior` or `posterior_descendant`."
            )

        self.model = model
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.epsilon = epsilon
        self.depth = max_tokens
        self.width = vocab_size if vocab_size is not None else model.config.vocab_size
        self.prior_kind = prior_kind
        self.prior_dir = prior_dir
        self.prior_dirichlet_alpha = prior_dirichlet_alpha
        self.prior_empirical_dataset_name = prior_empirical_dataset_name
        self.prior_empirical_llm_samples = prior_empirical_llm_samples
        self.sample_size = sample_size
        self.buffer_size = max_beam_size
        self.max_beam_size = max_beam_size
        self.used_max_beam_size = np.zeros(self.depth + 1)
        self.pruned_depth = -1
        self.device = next(model.parameters()).device
        self.stop_at_eos = stop_at_eos
        self.eos_token = self.model.config.eos_token_id
        self.acquisition_function = acquisition_function

        # For encoder-decoder/seq2seq models
        if self.is_encoder_decoder:
            tokens = torch.ones((1, 1), dtype=torch.long, device=self.device)
            tokens *= model.config.decoder_start_token_id
            self.encoder_inputs = model_inputs["input_ids"].to(self.device)
            # Cache encoder outputs since it is fixed.
            # (Used only to condition the generation process in the decoder.)
            self.encoder_outputs = model.encoder(**model_inputs.to(self.device))
        else:
            tokens = model_inputs["input_ids"].to(self.device)
            self.encoder_inputs = None
            self.encoder_outputs = None

        self.tree = nx.DiGraph()
        self.tree.add_node(
            "0",
            tokens=tokens,
            loglike=1,
            samples=np.ones(2),
            max_samples=np.ones(2),
            depth=0,
            active=True,
            best_child=None,
            explored=False,
            best_max_child=None,
        )
        self.betaparameters = torch.from_numpy(self.init_prior()).to(self.device)

    def init_prior(self) -> np.ndarray:
        """Build the approximate prior over Delta or load if already exists.

        Returns:
            beta_params: Float ndarray of shape (depth, 4). Where each level contains
                Beta's `(a, b, loc, scale)` parameters.
        """
        if self.prior_kind == "dirichlet":
            FNAME = f"{self.prior_dir}/prior_depth{self.depth}_width{self.width}_alpha{self.prior_dirichlet_alpha}.npy"
        else:
            FNAME = f"{self.prior_dir}/empirical_prior_depth{self.depth}_width{self.width}_dataset{self.prior_empirical_dataset_name}.npy"

        if os.path.isfile(FNAME):
            prior = np.load(FNAME)

            if isinstance(prior, torch.Tensor):
                prior = prior.numpy()

            return prior

        def sample():
            if self.prior_kind == "dirichlet":
                return np.random.dirichlet(
                    [self.prior_dirichlet_alpha] * self.width, size=self.sample_size
                )
            else:
                assert self.prior_empirical_llm_samples is not None
                n_emp_prior = self.prior_empirical_llm_samples.shape[0]
                return (
                    self.prior_empirical_llm_samples[torch.randperm(n_emp_prior)]
                    .cpu()
                    .numpy()
                )

        print("Cache not found. Computing the prior...")
        beta_params = np.ones((self.depth, 4))

        for d in tqdm.trange(self.depth):
            ps = sample()

            if d != 0:
                ps = sample()
                a, b, loc, scale = beta_params[d - 1, :]
                betas = np.asarray(
                    stats.beta.rvs(a, b, loc, scale, size=self.sample_size * self.width)
                )
                betas = betas.reshape(self.sample_size, self.width)
                ps = np.multiply(ps, betas)

            max_ps = np.max(ps, axis=1)
            a, b, loc, scale = stats.beta.fit(max_ps)

            beta_params[d, 0] = a
            beta_params[d, 1] = b
            beta_params[d, 2] = loc
            beta_params[d, 3] = scale

        if not os.path.exists(self.prior_dir):
            os.makedirs(self.prior_dir)

        np.save(FNAME, beta_params)

        return beta_params

    def winner_index(self, samples):
        """Compute winner index (i.e. index where max is obtained most often). Also return max."""
        max_samples, argmax_samples = torch.max(samples, dim=0)
        counts = torch.bincount(argmax_samples)
        return torch.argmax(counts), max_samples

    def recursive_best_child(self, node: str) -> str:
        """Recursively select the best child node (either based on posterior of node or posterior
        of best descendant).

        Args:
            node: The name of the starting (root) node.

        Returns:
            node: The name of the best node in the subtree.
        """
        if self.acquisition_function == "posterior":
            criterion = "best_max_child"
        else:
            criterion = "best_child"
        if self.tree.nodes[node][criterion]:
            return self.recursive_best_child(self.tree.nodes[node][criterion])
        children = list(self.tree.successors(node))
        if children:
            if self.acquisition_function == "posterior":
                max_children_samples = torch.stack(
                    [self.tree.nodes[child]["max_samples"] for child in children]
                )
                winner_index, _ = self.winner_index(max_children_samples)
            else:
                children_samples = torch.stack(
                    [self.tree.nodes[child]["samples"] for child in children]
                )
                winner_index, _ = self.winner_index(children_samples)
            return self.recursive_best_child(children[winner_index])
        self.tree.nodes[node]["explored"] = True
        return node

    def backup(self, node: str) -> None:
        """Update the distribution over the optimal value of the nodes by replacing it
        with the distribution of the optimal value of it's best child. (This is only a greedy
        approximation to the true distribution!). Alternative: Actual posterior in "max_samples"
        (this is the true distiribution for v!).

        Args:
            node: The name of the starting (root) node.
        """
        children = list(self.tree.successors(node))

        if self.acquisition_function == "posterior_descendant":
            children_samples = torch.stack(
                [self.tree.nodes[child]["samples"] for child in children]
            )
            winner_index, _ = self.winner_index(children_samples)
            best_child = children[winner_index]
            self.tree.nodes[node]["samples"] = self.tree.nodes[best_child]["samples"]
            self.tree.nodes[node]["best_child"] = best_child
        else:
            max_children_samples = torch.stack(
                [self.tree.nodes[child]["max_samples"] for child in children]
            )
            winner_index, max_samples = self.winner_index(max_children_samples)
            self.tree.nodes[node]["max_samples"] = max_samples
            self.tree.nodes[node]["best_max_child"] = children[winner_index]

        if not node == "0":
            parent = next(self.tree.predecessors(node))
            self.backup(parent)

    def budget_left(self) -> bool:
        """If the number of expanded nodes on the the last level exceeds the maximum
        number of nodes we can expand per level, there is no budget left anymore.

        Returns:
            is_budge_left: `True` if there is budget left, otherwise `False`.
        """
        return self.max_beam_size >= self.used_max_beam_size[-1]

    def set_nodes_to_inactive(self) -> None:
        """Check the number of expanded nodes per level of the tree. If this number
        exceeds the constraint on the maximal number, set all nodes on this level and
        the levels above to inactive.
        """
        new_first_level_nodes = [
            x
            for x, y in self.tree.nodes(data=True)
            if y["depth"] == self.pruned_depth and y["explored"]
        ]
        pruned_level_nodes = [
            x
            for x, y in self.tree.nodes(data=True)
            if y["depth"] < self.pruned_depth and y["depth"] != 0
        ]
        new_edges = [
            ("0", first_level_node) for first_level_node in new_first_level_nodes
        ]
        self.tree.add_edges_from(new_edges)
        self.tree.remove_nodes_from(list(n for n in pruned_level_nodes))
        self.backup("0")

    def predict(self, tokens: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the LLM. Returning the top-k best logprobs and indices.
        Here, `k` equals `max_beam_size`.

        Args:
            tokens: `torch.LongTensor` of shape (1, seq_len).

        Returns:
            top_k_probs: `torch.FloatTensor` of shape (k,) of top-k best probabilities.
            top_k_indices: `torch.LongTensor` of shape (k,) of top-k best children indices.
        """
        self.model.eval()
        tokens.to(self.device)

        with torch.no_grad():
            if self.is_encoder_decoder:
                outputs = self.model(
                    input_ids=self.encoder_inputs,
                    decoder_input_ids=tokens,
                    encoder_outputs=self.encoder_outputs,
                )
            else:
                outputs = self.model(input_ids=tokens)

            # Also see:
            # https://github.com/huggingface/transformers/blob/c54a8ca48eb1b85785f7fdbefb5311f172d19726/src/transformers/generation/logits_process.py#L225-L231
            if not self.stop_at_eos:
                scores_processed = outputs.logits.clone()
                vocab_tensor = torch.arange(
                    outputs.logits.shape[-1], device=outputs.logits.device
                )
                eos_token_mask = torch.isin(vocab_tensor, self.eos_token)
                scores_processed = torch.where(
                    eos_token_mask, -math.inf, outputs.logits
                )
                logprobs = torch.log_softmax(scores_processed, dim=-1)
            else:
                logprobs = torch.log_softmax(outputs.logits, dim=-1)

        nb_tokens = tokens.size(-1)
        old_logprobs = torch.sum(logprobs[0, range(nb_tokens - 1), tokens[0, 1:]])
        new_logprobs = old_logprobs + logprobs[0, -1, :]
        top_indices = torch.topk(new_logprobs, self.buffer_size).indices

        return new_logprobs[top_indices], top_indices

    def search(self) -> tuple[torch.Tensor, float, int]:
        """Main function for the search process.

        Returns:
            best_path: `torch.LongTensor` of shape (1, n_tokens).
            best_observed_value: Total logprob of the best path.
            n_llm_calls: Number of LLM forward passes done during the search.
        """
        best_path: torch.Tensor = torch.tensor(0).long()
        best_observed_value: float = -np.inf
        n_llm_calls: int = 0
        prob_result_nodes: float = 0

        # As long as the probability that we have found the maximum is too low and we
        # still have budget left keep searching:
        while (prob_result_nodes < 1 - self.epsilon) and self.budget_left():
            # Select a node for expansion
            new_node_name = self.recursive_best_child("0")
            new_node_tokens = self.tree.nodes[new_node_name]["tokens"]
            depth = self.tree.nodes[new_node_name]["depth"]

            # Is the maximum beam size for this level reached? If so deactivate
            # (i.e. prune) all nodes on this level and the levels above
            self.used_max_beam_size[depth] += 1

            if self.used_max_beam_size[depth] == self.max_beam_size:
                self.pruned_depth = depth
                self.set_nodes_to_inactive()

            # Add children (unless we are at the leaf level)
            if depth < self.depth and (
                new_node_tokens[0, -1] != self.eos_token or not self.stop_at_eos
            ):
                # predict the log likelihood for the children using the LLM
                n_llm_calls += 1
                child_depth = depth + 1
                children_observations, top_indices = self.predict(new_node_tokens)

                # generate samples from the prior for the optimal value of the remaining
                # token sequence for all children (that fit in the buffer)
                if child_depth == self.depth:
                    betas_all = torch.zeros(
                        (self.buffer_size, self.sample_size), device=self.device
                    )
                else:
                    a, b, loc, scale = self.betaparameters[self.depth - child_depth - 1]
                    m = Beta(a, b)
                    betas_all = (
                        m.sample(torch.Size([self.buffer_size, self.sample_size]))
                        * scale
                    ) + loc
                    betas_all = torch.log(betas_all)

                children_samples = children_observations[:, None] + betas_all

                children_tokens = torch.cat(
                    (new_node_tokens.repeat(self.buffer_size, 1), top_indices[:, None]),
                    dim=-1,
                )

                # Add the children to the tree
                for i in range(self.buffer_size):
                    child_obs = children_observations[i]
                    child_name = new_node_name + "*" + str(i)
                    child_tokens = children_tokens[i][None, :]

                    if self.stop_at_eos and child_tokens[0, -1] == self.eos_token:
                        child_samples = children_observations[i].repeat(
                            self.sample_size
                        )
                    else:
                        child_samples = children_samples[i]

                    self.tree.add_node(
                        child_name,
                        tokens=child_tokens,
                        loglike=child_obs,
                        samples=child_samples,
                        depth=child_depth,
                        active=True,
                        best_child=None,
                        explored=False,
                        max_samples=children_samples[i],
                        best_max_child=None,
                    )
                    self.tree.add_edge(new_node_name, child_name)

                    # If the child is a leaf node, add it to the result_nodes and
                    # potentially update the best observed result so far
                    if child_depth == self.depth or (
                        self.stop_at_eos and child_tokens[0, -1] == self.eos_token
                    ):
                        if child_obs > best_observed_value:
                            best_path = children_tokens[i][None, :]
                            best_observed_value = child_obs.item()

                # Update optimal value distribution of parents
                self.backup(new_node_name)

            # Update the estimate for the probability that we found the optimal path
            if self.acquisition_function == "posterior":
                overall_max_samples = self.tree.nodes["0"]["max_samples"]
            else:
                overall_max_samples = self.tree.nodes["0"]["samples"]
            prob_result_nodes = (
                torch.sum(best_observed_value >= overall_max_samples) / self.sample_size
            )

        return best_path, best_observed_value, n_llm_calls
