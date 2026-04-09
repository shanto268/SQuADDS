"""Heterogeneous GATv2 GNN for Universal Graph Pipeline.

Uses PyG HeteroData with typed nodes and edges, HeteroConv with typed
convolution operations, and LayerNorm + residual connections.

ALL nodes predict ALL Hamiltonian targets during training — the GNN learns
which design parameters affect which targets through message passing.
The target assignment metadata is used only at inference to know which
predictions to read from which node types.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

# ── Target definitions ────────────────────────────────────────────────
NODE_TARGET_NAMES = [
    "qubit_freq_GHz",
    "anharmonicity_MHz",
    "cavity_freq_GHz",
    "kappa_kHz",
    "g_MHz",
]
NUM_NODE_TARGETS = len(NODE_TARGET_NAMES)

# Inference readout map: which targets to READ from which component type.
# This is NOT used during training — all nodes learn all targets.
INFERENCE_READOUT = {
    "TransmonCross": ["qubit_freq_GHz", "anharmonicity_MHz", "g_MHz"],
    "Claw": ["g_MHz"],
    "RouteMeander": ["cavity_freq_GHz", "kappa_kHz", "g_MHz"],
    "CoupledLineTee": ["g_MHz"],
}


class UniversalGNN(nn.Module):
    """Heterogeneous GATv2 model with typed convolutions.

    All component nodes predict all 5 Hamiltonian targets.
    The model learns through message passing which nodes carry
    which information — no manual filtering.
    """

    def __init__(
        self,
        comp_dim: int,
        virt_dim: int,
        phys_edge_dim: int,
        spat_edge_dim: int,
        hidden_dim: int = 128,
        edge_hidden: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Input projections ─────────────────────────────────────────
        self.proj_comp = nn.Linear(comp_dim, hidden_dim)
        self.proj_virt = nn.Linear(virt_dim, hidden_dim)
        self.proj_phys_edge = nn.Linear(phys_edge_dim, edge_hidden)
        self.proj_spat_edge = nn.Linear(spat_edge_dim, edge_hidden)

        # ── HeteroConv layers ─────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("component", "physical", "component"): GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        edge_dim=edge_hidden,
                        add_self_loops=False,
                    ),
                    ("component", "spatial_in", "virtual"): SAGEConv(
                        hidden_dim,
                        hidden_dim,
                    ),
                    ("virtual", "spatial_out", "component"): GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        edge_dim=edge_hidden,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(
                nn.ModuleDict(
                    {
                        "component": nn.LayerNorm(hidden_dim),
                        "virtual": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        # ── Node prediction head (ALL nodes predict ALL targets) ──────
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_NODE_TARGETS),
        )

    def forward(self, data) -> dict:
        """Forward pass on HeteroData.

        Returns:
            dict with 'node_preds': Tensor [N_comp, NUM_NODE_TARGETS]
        """
        x_dict = {
            "component": self.proj_comp(data["component"].x),
            "virtual": self.proj_virt(data["virtual"].x),
        }

        edge_attr_dict = {}
        phys_key = ("component", "physical", "component")
        spat_key = ("virtual", "spatial_out", "component")

        if phys_key in data.edge_types:
            edge_attr_dict[phys_key] = self.proj_phys_edge(data[phys_key].edge_attr)
        if spat_key in data.edge_types:
            edge_attr_dict[spat_key] = self.proj_spat_edge(data[spat_key].edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)

            for node_type in x_dict:
                if node_type in x_dict_new:
                    x_dict[node_type] = norm[node_type](torch.relu(x_dict_new[node_type]) + x_dict[node_type])

        # ALL component nodes predict ALL targets
        node_preds = self.node_mlp(x_dict["component"])

        return {"node_preds": node_preds}
