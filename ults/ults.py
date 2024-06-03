from collections import UserDict
import os
import torch
from torch.distributions.beta import Beta
from scipy import stats
import numpy as np
import networkx as nx
import tqdm


class ULTS:
    """ULTS: Uncertainty-guided Likelihood-Tree Search.

    Args:
        model: A Huggingface LLM model.
        model_inputs: The input of `model(...)` or `model.forward(...)`. Must contain key "input_ids".
        max_tokens: Maximum number of tokens to generate.
        vocab_size: Vocabulary size. This should be your `tokenizer.vocab_size` or `len(tokenizer)`.
        max_beam_size: Maximum number of nodes to expand per level.
        epsilon: Confidence level for termination.
        prior_kind: "dirichlet" or "empirical".
        prior_dirichlet_alpha: Concentration parameter of the Dirichlet prior.
        prior_empirical_llm_samples: LLM output samples for the empirical prior.
        sample_size: Number of posterior samples to use.
        stop_at_eos: Consider sequences that end with <EOS> as leaf nodes.
        stop_overall: Use the distribution of the overall maximum to calculate the stopping
        condition or only the distirbution of the best next observation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_inputs: UserDict,
        max_tokens: int,
        vocab_size: int,
        max_beam_size: int = 5,
        epsilon: float = 0.1,
        prior_kind: str = "dirichlet",
        prior_dirichlet_alpha: float = 0.0001,
        prior_empirical_llm_samples: torch.Tensor | None = None,
        sample_size: int = 1000,
        stop_at_eos: bool = False,
        stop_overall: bool = False,
    ):
        if prior_kind == "empirical" and prior_empirical_llm_samples is None:
            raise ValueError(
                "`prior_empirical_llm_samples` cannot be `None` for empirical prior."
            )

        self.model = model
        self.model_inputs = model_inputs
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.epsilon = epsilon
        self.depth = max_tokens
        self.width = vocab_size
        self.prior_kind = prior_kind
        self.prior_dirichlet_alpha = prior_dirichlet_alpha
        self.prior_empirical_llm_samples = prior_empirical_llm_samples
        self.sample_size = sample_size
        self.vocab = torch.arange(start=0, end=vocab_size)
        self.buffer_size = max_beam_size
        self.max_beam_size = max_beam_size
        self.used_max_beam_size = np.zeros(self.depth + 1)
        self.pruned_depth = -1
        self.device = next(model.parameters()).device
        self.stop_at_eos = stop_at_eos
        self.eos_token = self.model.config.eos_token_id
        self.stop_overall = stop_overall

        # For encoder-decoder/seq2seq models
        if self.is_encoder_decoder:
            tokens = torch.ones((1, 1), dtype=torch.long, device=self.device)
            tokens *= model.config.decoder_start_token_id
            self.encoder_inputs = model_inputs["input_ids"].to(self.device)
            # Cache encoder outputs since it is fixed.
            # (Used only to condition the generation process in the decoder.)
            self.encoder_outputs = model.encoder(**model_inputs)
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
            max_samples = np.ones(2),
            depth=0,
            active=True,
            best_child=None,
            explored=False,
        )
        self.betaparameters = torch.from_numpy(self.init_prior()).to(self.device)

    def init_prior(self) -> torch.Tensor:
        """Build the approximate prior over Delta or load if already exists.

        Returns:
            beta_params: Ndarray of shape (depth, 4). Where each level contains
                Beta's `(a, b, loc, scale)` parameters.
        """
        DIRNAME = ".cache/priors"

        if self.prior_kind == "dirichlet":
            FNAME = f"{DIRNAME}/prior_depth{self.depth}_width{self.width}_alpha{self.prior_dirichlet_alpha}.npy"
        else:
            FNAME = f"{DIRNAME}/prior_depth{self.depth}_width{self.width}_emp.npy"

        if os.path.isfile(FNAME):
            return np.load(FNAME)

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

        if not os.path.exists(DIRNAME):
            os.makedirs(DIRNAME)

        np.save(FNAME, beta_params)

        return beta_params

    def recursive_best_child(self, node: str) -> str:
        """Recursively select the best child node (i.e. the node that has the higest probability
        to contain the maximum).

        Args:
            node: The name of the starting (root) node.

        Returns:
            node: The name of the best node in the subtree.
        """
        if self.tree.nodes[node]["best_child"]:
            return self.recursive_best_child(self.tree.nodes[node]["best_child"])

        children = list(self.tree.successors(node))

        if len(children) > 0:
            children_samples = torch.stack(
                [self.tree.nodes[child]["samples"] for child in children]
            )
            argmax_children_samples = torch.argmax(children_samples, dim=0).tolist()
            most_common_index = max(
                set(argmax_children_samples), key=argmax_children_samples.count
            )
            best_child = children[most_common_index]

            return self.recursive_best_child(best_child)

        self.tree.nodes[node]["explored"] = True

        return node

    def recursive_take_children_max(self, node: str) -> None:
        """Update the distribution over the optimal value of the nodes by replacing it
        with the distribution of the optimal value of it's best child. (This is only an
        approximation to the true distribution!)

        Args:
            node: The name of the starting (root) node.
        """
        children = list(self.tree.successors(node))
        children_samples = torch.stack(
            [self.tree.nodes[child]["samples"] for child in children]
        )
        max_samples = torch.max(children_samples, dim=0)[0]
        my_argmax_children_samples = torch.max(children_samples, dim=0)[1]
        my_counts = torch.bincount(my_argmax_children_samples)
        most_common_index = torch.argmax(my_counts)
        best_child = children[most_common_index]
        self.tree.nodes[node]["samples"] = self.tree.nodes[best_child]["samples"]
        self.tree.nodes[node]["best_child"] = best_child
        self.tree.nodes[node]["max_samples"] = max_samples

        if not node == "0":
            parent = next(self.tree.predecessors(node))
            self.recursive_take_children_max(parent)

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

            logprobs = torch.log_softmax(outputs.logits, dim=-1)

        nb_tokens = tokens.size(-1)
        old_logprobs = torch.sum(logprobs[0, range(nb_tokens - 1), tokens[0, 1:]])

        # Record context's loglik in the tree's root
        # Useful to cheaply compute the generated sequence's loglik since ULTS'
        # leaf record the *total* loglik (including the context's)
        if tokens.shape[-1] == self.model_inputs["input_ids"].shape[-1]:
            self.tree.nodes["0"]["loglike"] = old_logprobs.item()

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

                    self.tree.add_node(
                        child_name,
                        tokens=child_tokens,
                        loglike=child_obs,
                        samples=children_samples[i],
                        depth=child_depth,
                        active=True,
                        best_child=None,
                        explored=False,
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
            self.recursive_take_children_max(new_node_name)

            # Update the estimate for the probability that we found the optimal path
            if self.stop_overall:
                overall_max_samples = self.tree.nodes["0"]["max_samples"]
            else:
                overall_max_samples = self.tree.nodes["0"]["samples"]
            prob_result_nodes = (
                torch.sum(best_observed_value >= overall_max_samples) / self.sample_size
            )

        return best_path, best_observed_value, n_llm_calls
