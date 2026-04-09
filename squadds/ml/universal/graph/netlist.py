"""Circuit netlist dataclasses for universal graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Port:
    """A port on a quantum component."""

    name: str
    connection: str
    coupling_type: str  # "capacitive", "galvanic", "inductive", "open", "short", "microwave"


@dataclass
class ComponentSpec:
    """A quantum component specification."""

    name: str
    component_type: str  # "TransmonCross", "Claw", "RouteMeander", "CoupledLineTee"
    ports: dict[str, Port] = field(default_factory=dict)
    targets: list[str] = field(default_factory=list)  # Hamiltonian params predicted by this node


@dataclass
class EdgeSpec:
    """An edge between two quantum components."""

    src: str  # e.g., "qubit.N"
    dst: str  # e.g., "claw.pin"
    coupling_type: str
    targets: list[str] = field(default_factory=list)  # Hamiltonian params predicted by this edge


@dataclass
class CircuitNetlist:
    """A full quantum circuit netlist."""

    components: list[ComponentSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)

    def validate(self) -> None:
        """Validate the netlist structure."""
        comp_names = {comp.name for comp in self.components}
        for edge in self.edges:
            src_comp = edge.src.split(".")[0]
            dst_comp = edge.dst.split(".")[0]
            if src_comp not in comp_names:
                raise ValueError(f"Edge source component '{src_comp}' not found in netlist.")
            if dst_comp not in comp_names:
                raise ValueError(f"Edge destination component '{dst_comp}' not found in netlist.")

    def to_pyg_edge_index(self) -> torch.Tensor:
        """Convert netlist edges to a PyG edge_index tensor.

        Returns:
            torch.Tensor: Shape (2, E*2) where E is the number of undirected edges.
            Edges are treated as undirected, so both (src, dst) and (dst, src) are included.
        """
        comp_to_idx = {comp.name: i for i, comp in enumerate(self.components)}
        edge_list = []
        for edge in self.edges:
            src_idx = comp_to_idx[edge.src.split(".")[0]]
            dst_idx = comp_to_idx[edge.dst.split(".")[0]]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])  # Add reverse edge

        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
