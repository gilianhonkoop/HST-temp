from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GINConv


def _build_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Linear(in_dim, hidden_dim),
		nn.ReLU(),
		nn.Linear(hidden_dim, hidden_dim),
	)


class RandomPEARLEncoder(nn.Module):
	"""
	R-PEARL style positional encoder.

	Idea:
	  1) Initialize each node with M random samples.
	  2) Run the same message-passing network independently per sample.
	  3) Aggregate across samples (mean/sum) to obtain permutation-equivariant PE.

	This module only returns node positional encodings. You can concatenate the
	returned tensor with node features before feeding to downstream models
	(e.g., FraudGTConv).
	"""

	def __init__(
		self,
		hidden_dim: int = 128,
		num_phi_layers: int = 4,
		num_samples: int = 32,
		sample_pool: str = "mean",
		random_std: float = 1.0,
		random_seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if num_phi_layers < 1:
			raise ValueError("num_phi_layers must be >= 1")
		if num_samples < 1:
			raise ValueError("num_samples must be >= 1")
		if sample_pool not in {"mean", "sum"}:
			raise ValueError("sample_pool must be one of {'mean', 'sum'}")

		self.hidden_dim = hidden_dim
		self.num_phi_layers = num_phi_layers
		self.num_samples = num_samples
		self.sample_pool = sample_pool
		self.random_std = random_std
		self.random_seed = int(random_seed) if random_seed is not None else None

		self.input_proj = nn.Linear(1, hidden_dim)
		self.phi_layers = nn.ModuleList(
			[GINConv(_build_mlp(hidden_dim, hidden_dim)) for _ in range(num_phi_layers)]
		)
		self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_phi_layers)])

	def sample_random_inputs(
		self,
		num_nodes: int,
		device: torch.device,
		dtype: torch.dtype = torch.float,
		num_edges: int = 0,
	) -> Tensor:
		"""Create random node inputs of shape [N, M, 1]."""
		if self.random_seed is None:
			return torch.randn(num_nodes, self.num_samples, 1, device=device, dtype=dtype) * self.random_std

		# Use a local generator so PEARL encodings are deterministic for a given
		# seed and graph shape, independent of global RNG state or call order.
		seed = (self.random_seed + 1_000_003 * num_nodes + 9_176 * num_edges) % (2**63 - 1)
		generator = torch.Generator(device=device)
		generator.manual_seed(seed)
		return torch.randn(
			num_nodes,
			self.num_samples,
			1,
			device=device,
			dtype=dtype,
			generator=generator,
		) * self.random_std

	def _run_phi(self, x: Tensor, edge_index: Tensor) -> Tensor:
		"""Run shared phi network for one sample input, shape [N, H]."""
		h = x
		for conv, norm in zip(self.phi_layers, self.norms):
			h = conv(h, edge_index)
			if h.size(0) > 1:
				h = norm(h)
			h = torch.relu(h)
		return h

	def forward(self, data: Data, rand: Optional[Tensor] = None) -> Tensor:
		"""
		Args:
			data: PyG Data with at least `edge_index` and `num_nodes`.
			rand: Optional random tensor with shape [N, M] or [N, M, 1].
				  If None, random samples are generated internally.

		Returns:
			Positional encodings of shape [N, hidden_dim].
		"""
		if data.num_nodes is None:
			raise ValueError("data.num_nodes is required for RandomPEARLEncoder")

		num_nodes = data.num_nodes
		edge_index = data.edge_index
		data_x = getattr(data, "x", None)
		device = edge_index.device if edge_index is not None else (data_x.device if data_x is not None else torch.device("cpu"))
		num_edges = int(edge_index.size(1)) if edge_index is not None else 0

		if rand is None:
			rand = self.sample_random_inputs(num_nodes=num_nodes, device=device, num_edges=num_edges)
		else:
			if rand.dim() == 2:
				rand = rand.unsqueeze(-1)
			if rand.dim() != 3 or rand.size(0) != num_nodes or rand.size(2) != 1:
				raise ValueError("rand must have shape [N, M] or [N, M, 1]")

		m_samples = rand.size(1)
		h0 = self.input_proj(rand)  # [N, M, H]

		# If there are no edges, return projected pooled random inputs.
		if edge_index is None or edge_index.numel() == 0:
			return h0.mean(dim=1) if self.sample_pool == "mean" else h0.sum(dim=1)

		# Vectorize over samples: reshape to [N*M, H] and create batched edge_index
		h0_batch = h0.view(-1, self.hidden_dim)
		
		edge_indices = [edge_index + i * num_nodes for i in range(m_samples)]
		edge_index_batch = torch.cat(edge_indices, dim=1)

		# Run GIN on the batched graph
		h_batch = self._run_phi(h0_batch, edge_index_batch)

		# Reshape back to [N, M, H]
		h_all = h_batch.view(num_nodes, m_samples, self.hidden_dim)

		if self.sample_pool == "mean":
			return h_all.mean(dim=1)
		return h_all.sum(dim=1)
