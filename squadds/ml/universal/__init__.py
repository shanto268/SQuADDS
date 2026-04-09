"""Universal Graph Neural Network pipeline for quantum chip design.

This module provides a tool-agnostic ML pipeline that:
1. Generates Shapely polygon geometries from parametric design dictionaries
2. Extracts geometric features (moments, rasterized masks, parameter embeddings)
3. Constructs PyG graphs with virtual hub nodes
4. Predicts Hamiltonian parameters via GATv2 with node/edge prediction heads
"""
