"""Heterogeneous GATv2 GNN for Universal Graph Pipeline.

ALL nodes predict ALL node targets. ALL physical edges predict ALL edge targets.
The GNN learns which parameters affect which targets through message passing.
INFERENCE_READOUT maps are metadata for reading predictions at inference.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

# ── Target definitions ────────────────────────────────────────────────
# Node targets: intrinsic to components
NODE_TARGET_NAMES = ["qubit_freq_GHz", "anharmonicity_MHz", "cavity_freq_GHz"]
NUM_NODE_TARGETS = len(NODE_TARGET_NAMES)

# Edge targets: arise from interactions between components
EDGE_TARGET_NAMES = ["g_MHz", "kappa_kHz"]
NUM_EDGE_TARGETS = len(EDGE_TARGET_NAMES)

# Inference readout: which targets to READ from which component type.
# NOT used during training.
NODE_INFERENCE_READOUT = {
    "TransmonCross": ["qubit_freq_GHz", "anharmonicity_MHz"],
    "Claw": [],
    "RouteMeander": ["cavity_freq_GHz"],
    "CoupledLineTee": [],
}

# Edge readout: which targets to READ from which edge type.
EDGE_INFERENCE_READOUT = {
    ("TransmonCross", "Claw"): ["g_MHz"],
    ("Claw", "TransmonCross"): ["g_MHz"],
    ("RouteMeander", "CoupledLineTee"): ["kappa_kHz"],
    ("CoupledLineTee", "RouteMeander"): ["kappa_kHz"],
}


class UniversalGNN(nn.Module):
    """Heterogeneous GATv2 model with typed convolutions.

    Predicts:
    - Node targets: qubit_freq, anharmonicity, cavity_freq (per component node)
    - Edge targets: g, kappa (per physical edge)
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
        self.edge_hidden = edge_hidden

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

        # ── Node prediction head ──────────────────────────────────────
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_NODE_TARGETS),
        )

        # ── Edge prediction head ──────────────────────────────────────
        # Concat src + dst node embeddings + projected edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_EDGE_TARGETS),
        )

    def forward(self, data) -> dict:
        """Forward pass on HeteroData.

        Returns:
            dict with:
                'node_preds': Tensor [N_comp, NUM_NODE_TARGETS]
                'edge_preds': Tensor [E_phys, NUM_EDGE_TARGETS]
        """
        x_dict = {
            "component": self.proj_comp(data["component"].x),
            "virtual": self.proj_virt(data["virtual"].x),
        }

        edge_attr_dict = {}
        phys_key = ("component", "physical", "component")
        spat_key = ("virtual", "spatial_out", "component")

        phys_edge_proj = None
        if phys_key in data.edge_types:
            phys_edge_proj = self.proj_phys_edge(data[phys_key].edge_attr)
            edge_attr_dict[phys_key] = phys_edge_proj
        if spat_key in data.edge_types:
            edge_attr_dict[spat_key] = self.proj_spat_edge(data[spat_key].edge_attr)

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            for node_type in x_dict:
                if node_type in x_dict_new:
                    x_dict[node_type] = norm[node_type](torch.relu(x_dict_new[node_type]) + x_dict[node_type])

        # Node predictions
        node_preds = self.node_mlp(x_dict["component"])

        # Edge predictions (physical edges only)
        if phys_key in data.edge_types and phys_edge_proj is not None:
            edge_index = data[phys_key].edge_index
            src, dst = edge_index
            edge_input = torch.cat(
                [x_dict["component"][src], x_dict["component"][dst], phys_edge_proj],
                dim=-1,
            )
            edge_preds = self.edge_mlp(edge_input)
        else:
            edge_preds = torch.empty(0, NUM_EDGE_TARGETS)

        return {"node_preds": node_preds, "edge_preds": edge_preds}
