"""Heterogeneous GATv2 Graph Neural Network for Universal Graph Pipeline.

Uses PyG HeteroData with typed nodes ('component', 'virtual') and typed edges
('physical', 'spatial_in', 'spatial_out'). Each edge type gets its own
convolution operation via HeteroConv.

Target predictions are node-type-aware: only components predict Hamiltonian
parameters, and each component only predicts the targets relevant to its type.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

# ── Target registry: which component types predict which targets ──────
# Node targets indexed as: [qubit_freq, anharmonicity, cavity_freq, kappa]
NODE_TARGET_NAMES = ["qubit_freq_GHz", "anharmonicity_MHz", "cavity_freq_GHz", "kappa_kHz"]
EDGE_TARGET_NAMES = ["g_MHz"]

COMPONENT_TARGET_MASK = {
    "TransmonCross": [True, True, False, False],  # qubit_freq, anharmonicity
    "Claw": [False, False, False, False],  # no direct predictions
    "RouteMeander": [False, False, True, True],  # cavity_freq, kappa
    "CoupledLineTee": [False, False, False, False],  # no direct predictions
}

NUM_NODE_TARGETS = len(NODE_TARGET_NAMES)
NUM_EDGE_TARGETS = len(EDGE_TARGET_NAMES)


class UniversalGNN(nn.Module):
    """Heterogeneous GATv2 model with typed convolutions.

    Architecture:
    - Input projection layers for each node/edge type
    - K layers of HeteroConv with:
        - GATv2Conv for component <-> component (physical)
        - SAGEConv for component -> virtual (spatial pooling)
        - GATv2Conv for virtual -> component (spatial broadcast)
    - Node prediction head (per component node)
    - Edge prediction head (per physical edge)
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
        """
        Args:
            comp_dim: Dimension of component node features.
            virt_dim: Dimension of virtual node features.
            phys_edge_dim: Dimension of physical edge features.
            spat_edge_dim: Dimension of spatial edge features.
            hidden_dim: Uniform hidden dimension after projection.
            edge_hidden: Compressed edge dimension for attention.
            num_layers: Number of HeteroConv layers.
            num_heads: Number of attention heads.
        """
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

        for _i in range(num_layers):
            conv = HeteroConv(
                {
                    # Component <-> Component (physical micro-exchange)
                    ("component", "physical", "component"): GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        edge_dim=edge_hidden,
                        add_self_loops=False,
                    ),
                    # Component -> Virtual (macro-pooling)
                    ("component", "spatial_in", "virtual"): SAGEConv(hidden_dim, hidden_dim),
                    # Virtual -> Component (macro-broadcast)
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

            # LayerNorm per node type
            self.norms.append(
                nn.ModuleDict(
                    {
                        "component": nn.LayerNorm(hidden_dim),
                        "virtual": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        # ── Prediction heads ──────────────────────────────────────────
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_NODE_TARGETS),
        )

        # Edge prediction: concat src + dst + edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_EDGE_TARGETS),
        )

    def forward(self, data) -> dict:
        """Forward pass on HeteroData.

        Args:
            data: PyG HeteroData with node types 'component' and 'virtual',
                  and edge types 'physical', 'spatial_in', 'spatial_out'.

        Returns:
            dict with:
                'node_preds': Tensor [N_comp, NUM_NODE_TARGETS]
                'edge_preds': Tensor [E_phys, NUM_EDGE_TARGETS]
        """
        # ── Project inputs ────────────────────────────────────────────
        x_dict = {
            "component": self.proj_comp(data["component"].x),
            "virtual": self.proj_virt(data["virtual"].x),
        }

        # Project edge features
        edge_attr_dict = {}
        if ("component", "physical", "component") in data.edge_types:
            edge_attr_dict[("component", "physical", "component")] = self.proj_phys_edge(
                data["component", "physical", "component"].edge_attr
            )
        if ("virtual", "spatial_out", "component") in data.edge_types:
            edge_attr_dict[("virtual", "spatial_out", "component")] = self.proj_spat_edge(
                data["virtual", "spatial_out", "component"].edge_attr
            )

        # ── Message passing ───────────────────────────────────────────
        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(
                x_dict,
                data.edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            # ReLU + LayerNorm + residual
            for node_type in x_dict:
                if node_type in x_dict_new:
                    x_dict[node_type] = norm[node_type](torch.relu(x_dict_new[node_type]) + x_dict[node_type])

        # ── Node predictions (component nodes only) ───────────────────
        node_preds = self.node_mlp(x_dict["component"])

        # ── Edge predictions (physical edges only) ────────────────────
        if ("component", "physical", "component") in data.edge_types:
            edge_index = data["component", "physical", "component"].edge_index
            src, dst = edge_index
            phys_edge_attr = edge_attr_dict.get(
                ("component", "physical", "component"),
                torch.zeros(edge_index.size(1), 1),
            )
            edge_input = torch.cat(
                [x_dict["component"][src], x_dict["component"][dst], phys_edge_attr],
                dim=-1,
            )
            edge_preds = self.edge_mlp(edge_input)
        else:
            edge_preds = torch.empty(0, NUM_EDGE_TARGETS)

        return {"node_preds": node_preds, "edge_preds": edge_preds}
