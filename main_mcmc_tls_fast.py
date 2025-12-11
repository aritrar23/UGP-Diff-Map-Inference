from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import random
import csv
import json
import math, random
import itertools
from typing import Iterable,  Dict, Tuple, List, Optional, Set, FrozenSet, Any
from collections import Counter, defaultdict
from tqdm import trange
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST BE CALLED BEFORE importing pyplot
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors
import arviz as az

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAP structure search for Carta-CDMIP model.

Score(F) = log P(F) + sum_T log P(T|F),
where P(T|F) = sum_{labelings} B(O+1, D+1), with per-node counts:
  obs = |L ∩ B(v)|, miss = |L \ B(v)|, and B(·) is Beta function.
p ~ Beta(1,1) is integrated out exactly.

- Newick parser (no external deps)
- DP over labelings with (O,D) sparse tables
- Priors: fixed-k (uniform over potency sets) OR Bernoulli(pi_P); edges Bernoulli(rho)
- Stochastic hill-climb + simulated annealing over F=(Z,A)
"""

# ==============================================================================
# === NEW EDGE SELECTION FUNCTIONS (for Phase 4) ===============================
# ==============================================================================

class UnionFind:
    """A simple Union-Find data structure for Kruskal's algorithm."""
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def kruskal_mst(nodes: Set[FrozenSet[str]], weighted_edges: List[Tuple[float, Tuple]]) -> Set[Tuple]:
    """
    Runs Kruskal's algorithm to find a MAXIMUM Spanning Tree.
    
    Args:
        nodes: All nodes (potencies) in the graph.
        weighted_edges: List of (weight, (P, Q)) tuples.
    
    Returns:
        A set of edges (P, Q) that form the MST.
    """
    mst_edges = set()
    uf = UnionFind(nodes)
    
    # Sort by weight DESCENDING to get a Maximum ST
    weighted_edges.sort(key=lambda x: x[0], reverse=True)
    
    for weight, (P, Q) in weighted_edges:
        if uf.union(P, Q):
            # This edge connects two new components
            mst_edges.add((P, Q))
            
    return mst_edges

# def build_A_map_from_flow(
#     viterbi_flow: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float],
#     Z_map: Set[FrozenSet[str]],
#     S_nodes: Set[FrozenSet[str]], # Set of singletons
#     z_score_threshold: float = 1.5 # This argument is no longer used, but kept for compatibility
# ) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]:
#     """
#     Builds the final A_map using the:
#     1. "Max-Flow Backbone" (MST)
#     2. "Branching Fix" (>= 2 children)
#     3. "20% Flow Rule" (Paper's heuristic)
#     """
#     print("Building final A_map from Viterbi flow...")
#     A_map: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int] = {}
    
#     if not viterbi_flow:
#         print("Warning: Viterbi flow is empty. Returning empty edge set.")
#         return {}

#     # --- 1. Step 4a: Find the "Max-Flow Backbone" (MST) ---
#     # (This section is UNCHANGED)
    
#     undirected_weights: Dict[FrozenSet[Tuple], float] = defaultdict(float)
#     for (P, Q), flow in viterbi_flow.items():
#         edge_key = frozenset([P, Q])
#         if flow > undirected_weights[edge_key]:
#             undirected_weights[edge_key] = flow
    
#     kruskal_edge_list = []
#     for edge_key, weight in undirected_weights.items():
#         if weight > 0:
#             P, Q = tuple(edge_key)
#             kruskal_edge_list.append((weight, (P, Q)))

#     print(f"Finding Maximum Spanning Tree (Backbone) from {len(kruskal_edge_list)} weighted edges...")
#     mst_edges_undir = kruskal_mst(Z_map, kruskal_edge_list)
    
#     print(f"Backbone (from MST) contains {len(mst_edges_undir)} directed edges:")
#     for (P, Q) in mst_edges_undir:
#         flow_PQ = viterbi_flow.get((P, Q), 0)
#         flow_QP = viterbi_flow.get((Q, P), 0)
        
#         edge_to_add = (P, Q) if flow_PQ >= flow_QP else (Q, P)
#         A_map[edge_to_add] = 1
#         print(f"  [MST] Added: {pot_str(edge_to_add[0])} -> {pot_str(edge_to_add[1])} (w={max(flow_PQ, flow_QP):.2f})")


#     # --- 2. Step 4b: Enforce >= 2 Children Constraint ---
#     # (This section is UNCHANGED)
    
#     print("Enforcing >= 2 children constraint for progenitors...")
#     progenitor_nodes = Z_map - S_nodes
#     added_for_branching = 0
    
#     for P in progenitor_nodes:
#         current_out_degree = sum(1 for (parent, child) in A_map if parent == P)
        
#         needed = 2 - current_out_degree
#         if needed <= 0:
#             continue
            
#         missing_edges = []
#         for Q in Z_map:
#             edge = (P, Q)
#             if P != Q and edge not in A_map and viterbi_flow.get(edge, 0) > 0:
#                 missing_edges.append((viterbi_flow[edge], edge))
        
#         missing_edges.sort(key=lambda x: x[0], reverse=True)
        
#         edges_to_add = missing_edges[:needed]
        
#         if edges_to_add:
#             print(f"  Fixing {pot_str(P)} (needs {needed} more):")
#             for flow, edge in edges_to_add:
#                 A_map[edge] = 1
#                 added_for_branching += 1
#                 print(f"    [BRANCH] Added: {pot_str(edge[0])} -> {pot_str(edge[1])} (w={flow:.2f})")
            
#     print(f"Added {added_for_branching} edges to satisfy branching constraint.")

#     # --- 3. Step 4c: Add Secondary Paths (Paper's 20% Rule) ---
#     # (This section is NEW and replaces the z-score logic)
    
#     print("Adding secondary edges with > 20% of parent's total flow...")

#     # First, pre-calculate the total *positive* Viterbi flow out of *every* node
#     total_flow_out = defaultdict(float)
#     for (P, Q), flow in viterbi_flow.items():
#         if flow > 0:
#             total_flow_out[P] += flow
            
#     added_for_flow = 0
    
#     # Now, check every potential edge that passed the Viterbi step
#     for (P, Q), flow in viterbi_flow.items():
#         edge = (P, Q)
        
#         # Check 1: Is it a positive-flow edge?
#         if flow <= 0:
#             continue
            
#         # Check 2: Is it *already* in our map? (from MST or branching)
#         if edge in A_map:
#             continue
            
#         # Check 3: Is the parent a progenitor? (We don't care about flow from singletons)
#         if P in S_nodes:
#             continue
            
#         # Check 4: Does it meet the 20% threshold?
#         parent_total_flow = total_flow_out.get(P, 0)
        
#         # Avoid division by zero, and apply the rule
#         if parent_total_flow > 0 and (flow / parent_total_flow) > 0.20:
#             A_map[edge] = 1
#             added_for_flow += 1
#             print(f"  [FLOW 20%] Added: {pot_str(P)} -> {pot_str(Q)} (w={flow:.2f} is > 20% of {parent_total_flow:.2f})")
            
#     print(f"Added {added_for_flow} edges satisfying the 20% flow rule.")
#     print(f"Final A_map contains {len(A_map)} total edges.")
    
#     return A_map

# def build_A_map_from_flow(
#     viterbi_flow: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float],
#     Z_map: Set[FrozenSet[str]],
#     S_nodes: Set[FrozenSet[str]], # Set of singletons
#     z_score_threshold: float = 1.5 # This argument is no longer used, but kept for compatibility
# ) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]:
#     """
#     Builds the final A_map using the:
#     1. "Max-Flow Backbone" (MST)
#     2. "Fitch Property Fix" (Union of children == Parent)
#     3. "20% Flow Rule" (Paper's heuristic)
#     """
#     print("Building final A_map from Viterbi flow...")
#     A_map: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int] = {}
    
#     if not viterbi_flow:
#         print("Warning: Viterbi flow is empty. Returning empty edge set.")
#         return {}

#     # --- 1. Step 4a: Find the "Max-Flow Backbone" (MST) ---
#     # (This section is UNCHANGED)
    
#     undirected_weights: Dict[FrozenSet[Tuple], float] = defaultdict(float)
#     for (P, Q), flow in viterbi_flow.items():
#         edge_key = frozenset([P, Q])
#         if flow > undirected_weights[edge_key]:
#             undirected_weights[edge_key] = flow
    
#     kruskal_edge_list = []
#     for edge_key, weight in undirected_weights.items():
#         if weight > 0:
#             P, Q = tuple(edge_key)
#             kruskal_edge_list.append((weight, (P, Q)))

#     print(f"Finding Maximum Spanning Tree (Backbone) from {len(kruskal_edge_list)} weighted edges...")
#     mst_edges_undir = kruskal_mst(Z_map, kruskal_edge_list)
    
#     print(f"Backbone (from MST) contains {len(mst_edges_undir)} directed edges:")
#     for (P, Q) in mst_edges_undir:
#         flow_PQ = viterbi_flow.get((P, Q), 0)
#         flow_QP = viterbi_flow.get((Q, P), 0)
        
#         edge_to_add = (P, Q) if flow_PQ >= flow_QP else (Q, P)
#         A_map[edge_to_add] = 1
#         print(f"  [MST] Added: {pot_str(edge_to_add[0])} -> {pot_str(edge_to_add[1])} (w={max(flow_PQ, flow_QP):.2f})")


#     # --- 2. Step 4b: Enforce Fitch (Union) Property ---
#     # (This section is NEW and replaces the ">= 2 children" logic)
    
#     print("Enforcing Fitch (Union) property for progenitors...")
#     progenitor_nodes = Z_map - S_nodes
#     added_for_fitch = 0
    
#     for P in progenitor_nodes:
#         # Get current children and their union
#         current_children = {Q for (p, Q) in A_map if p == P}
#         if not current_children:
#             current_union = frozenset()
#         else:
#             current_union = frozenset().union(*current_children)
            
#         if current_union == P:
#             continue # Property already satisfied

#         # We are missing fates
#         missing_fates = P - current_union
#         fates_still_missing = set(missing_fates) # mutable copy
#         print(f"  Fixing {pot_str(P)} (missing {pot_str(fates_still_missing)}):")

#         # Find all candidate edges that provide *any* of the original missing fates
#         candidate_edges = []
#         for Q in Z_map:
#             edge = (P, Q)
#             if P != Q and edge not in A_map:
#                 flow = viterbi_flow.get(edge, 0)
#                 # Check which of the *originally* missing fates this Q provides
#                 fates_provided = Q & missing_fates 
                
#                 if flow > 0 and len(fates_provided) > 0:
#                     candidate_edges.append((flow, edge, fates_provided))
                    
#         # Sort candidates by flow, descending
#         candidate_edges.sort(key=lambda x: x[0], reverse=True)
        
#         # Greedily add edges until the set of missing fates is empty
#         for flow, edge, fates_provided in candidate_edges:
#             if not fates_still_missing:
#                 break # We're done for this parent P
            
#             # Check if this edge *still* provides something we need
#             newly_provided = fates_provided & fates_still_missing
            
#             if newly_provided:
#                 A_map[edge] = 1
#                 added_for_fitch += 1
#                 fates_still_missing.difference_update(newly_provided)
#                 print(f"    [FITCH] Added: {pot_str(edge[0])} -> {pot_str(edge[1])} (w={flow:.2f}, provides {pot_str(newly_provided)})")
        
#         if fates_still_missing:
#             print(f"    [WARN] Could not satisfy Fitch for {pot_str(P)}. Still missing {pot_str(fates_still_missing)}.")

#     print(f"Added {added_for_fitch} edges to satisfy Fitch property.")


#     # --- 3. Step 4c: Add Secondary Paths (Paper's 20% Rule) ---
#     # (This section is UNCHANGED)
    
#     print("Adding secondary edges with > 20% of parent's total flow...")

#     total_flow_out = defaultdict(float)
#     for (P, Q), flow in viterbi_flow.items():
#         if flow > 0:
#             total_flow_out[P] += flow
            
#     added_for_flow = 0
    
#     for (P, Q), flow in viterbi_flow.items():
#         edge = (P, Q)
        
#         if flow <= 0:
#             continue
#         if edge in A_map: # Check if already added by MST or Fitch-fix
#             continue
#         if P in S_nodes: # Don't care about flow from singletons
#             continue
            
#         parent_total_flow = total_flow_out.get(P, 0)
        
#         if parent_total_flow > 0 and (flow / parent_total_flow) > 0.20:
#             A_map[edge] = 1
#             added_for_flow += 1
#             print(f"  [FLOW 20%] Added: {pot_str(P)} -> {pot_str(Q)} (w={flow:.2f} is > 20% of {parent_total_flow:.2f})")
            
#     print(f"Added {added_for_flow} edges satisfying the 20% flow rule.")
#     print(f"Final A_map contains {len(A_map)} total edges.")
    
#     return A_map

def build_A_map_from_flow(
    viterbi_flow: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float],
    Z_map: Set[FrozenSet[str]],
    S_nodes: Set[FrozenSet[str]], # Set of singletons
    z_score_threshold: float = 1.5 
) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]:
    """
    Builds the final A_map using the:
    1. "Greedy Best-Parent Backbone" (Guarantees directed reachability)
    2. "Fitch Property Fix" (Union of children == Parent)
    3. "20% Flow Rule" (Secondary paths)
    """
    print("Building final A_map from Viterbi flow...")
    A_map: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int] = {}
    
    if not viterbi_flow:
        print("Warning: Viterbi flow is empty. Returning empty edge set.")
        return {}

    # --- 1. Step 4a: Max-Flow Backbone (Greedy Best Parent) ---
    # Goal: Ensure every node (except root) has the single strongest incoming edge.
    
    # Identify the global root (largest set)
    # Sort just to be deterministic in case of ties
    sorted_nodes = sorted(list(Z_map), key=lambda x: (len(x), tuple(sorted(list(x)))), reverse=True)
    global_root = sorted_nodes[0]
    
    print(f"Identified Global Root: {pot_str(global_root)}")
    print("Constructing Arborescence (Best-Parent Backbone)...")

    for child in sorted_nodes:
        if child == global_root:
            continue
            
        # Find all valid parents for this child
        best_parent = None
        max_flow = -1.0
        
        # Check all possible parents in Z_map
        for potential_parent in Z_map:
            edge = (potential_parent, child)
            
            # Check if this edge exists in our Viterbi flow
            # (Note: viterbi_flow only contains admissible edges P->Q where Q is subset of P)
            if edge in viterbi_flow:
                flow = viterbi_flow[edge]
                if flow > max_flow:
                    max_flow = flow
                    best_parent = potential_parent
        
        if best_parent:
            edge = (best_parent, child)
            A_map[edge] = 1
            print(f"  [BACKBONE] Child {pot_str(child)} connected to {pot_str(best_parent)} (w={max_flow:.2f})")
        else:
            print(f"  [WARN] Child {pot_str(child)} has no flow from any parent! It is disconnected.")

    print(f"Backbone contains {len(A_map)} edges.")


    # --- 2. Step 4b: Enforce Fitch (Union) Property ---
    # (This section is UNCHANGED)
    
    print("Enforcing Fitch (Union) property for progenitors...")
    progenitor_nodes = Z_map - S_nodes
    added_for_fitch = 0
    
    for P in progenitor_nodes:
        # Get current children and their union
        current_children = {Q for (p, Q) in A_map if p == P}
        if not current_children:
            current_union = frozenset()
        else:
            current_union = frozenset().union(*current_children)
            
        if current_union == P:
            continue # Property already satisfied

        # We are missing fates
        missing_fates = P - current_union
        fates_still_missing = set(missing_fates) # mutable copy
        print(f"  Fixing {pot_str(P)} (missing {pot_str(fates_still_missing)}):")

        # Find all candidate edges that provide *any* of the original missing fates
        candidate_edges = []
        for Q in Z_map:
            edge = (P, Q)
            if P != Q and edge not in A_map:
                flow = viterbi_flow.get(edge, 0)
                # Check which of the *originally* missing fates this Q provides
                fates_provided = Q & missing_fates 
                
                if flow > 0 and len(fates_provided) > 0:
                    candidate_edges.append((flow, edge, fates_provided))
                    
        # Sort candidates by flow, descending
        candidate_edges.sort(key=lambda x: x[0], reverse=True)
        
        # Greedily add edges until the set of missing fates is empty
        for flow, edge, fates_provided in candidate_edges:
            if not fates_still_missing:
                break # We're done for this parent P
            
            # Check if this edge *still* provides something we need
            newly_provided = fates_provided & fates_still_missing
            
            if newly_provided:
                A_map[edge] = 1
                added_for_fitch += 1
                fates_still_missing.difference_update(newly_provided)
                print(f"    [FITCH] Added: {pot_str(edge[0])} -> {pot_str(edge[1])} (w={flow:.2f}, provides {pot_str(newly_provided)})")
        
        if fates_still_missing:
            print(f"    [WARN] Could not satisfy Fitch for {pot_str(P)}. Still missing {pot_str(fates_still_missing)}.")

    print(f"Added {added_for_fitch} edges to satisfy Fitch property.")


    # --- 3. Step 4c: Add Secondary Paths (Paper's 20% Rule) ---
    # (This section is UNCHANGED)
    
    print("Adding secondary edges with > 20% of parent's total flow...")

    total_flow_out = defaultdict(float)
    for (P, Q), flow in viterbi_flow.items():
        if flow > 0:
            total_flow_out[P] += flow
            
    added_for_flow = 0
    
    for (P, Q), flow in viterbi_flow.items():
        edge = (P, Q)
        
        if flow <= 0:
            continue
        if edge in A_map: # Check if already added
            continue
        if P in S_nodes: # Don't care about flow from singletons
            continue
            
        parent_total_flow = total_flow_out.get(P, 0)
        
        if parent_total_flow > 0 and (flow / parent_total_flow) > 0.20:
            A_map[edge] = 1
            added_for_flow += 1
            print(f"  [FLOW 20%] Added: {pot_str(P)} -> {pot_str(Q)} (w={flow:.2f} is > 20% of {parent_total_flow:.2f})")
            
    print(f"Added {added_for_flow} edges satisfying the 20% flow rule.")
    print(f"Final A_map contains {len(A_map)} total edges.")
    
    return A_map

def edges_from_A(A: Dict[Tuple[frozenset, frozenset], int]) -> Set[Tuple[frozenset, frozenset]]:
    return {e for e, v in A.items() if v == 1}

def jaccard_distance_edges(E1: Set[Tuple[frozenset, frozenset]], 
                           E2: Set[Tuple[frozenset, frozenset]]) -> float:
    if not E1 and not E2:
        return 0.0
    inter = len(E1 & E2)
    union = len(E1 | E2)
    return 1.0 - (inter / union if union else 1.0)

def _laplacian_eigs_undirected(nodes: List[frozenset],
                               edges: Set[Tuple[frozenset, frozenset]]) -> np.ndarray:
    """Undirect the edge set, build L = D - A over `nodes`, return eigenvalues (sorted)."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        i, j = idx[u], idx[v]
        if i == j:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0  # undirected
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    # symmetric PSD -> use eigvalsh
    vals = np.linalg.eigvalsh(L)
    return np.sort(vals)

def _spectral_density(omegas: np.ndarray, wgrid: np.ndarray, gamma: float) -> np.ndarray:
    """
    Lorentzian kernel density on frequencies (ω_i = sqrt(λ_i)).
    ρ(ω) = (1/n) * Σ_i [ γ / ( (ω - ω_i)^2 + γ^2 ) ]
    """
    if len(omegas) == 0:
        return np.zeros_like(wgrid)
    # broadcast: (m grid, k eigs)
    diff = wgrid[:, None] - omegas[None, :]
    dens = gamma / (diff * diff + gamma * gamma)
    return np.mean(dens, axis=1)

def _im_distance_from_spectra(lams1: np.ndarray, lams2: np.ndarray, gamma: float = 0.08) -> float:
    """
    Numerically integrate the L2 distance between spectral densities.
    Uses ω = sqrt(λ) grid up to the max observed plus a margin.
    """
    w1 = np.sqrt(np.maximum(lams1, 0.0))
    w2 = np.sqrt(np.maximum(lams2, 0.0))
    wmax = float(max(w1.max() if w1.size else 0.0, w2.max() if w2.size else 0.0, 1.0))
    wgrid = np.linspace(0.0, wmax + 3.0 * gamma, 2000)  # dense grid
    rho1 = _spectral_density(w1, wgrid, gamma)
    rho2 = _spectral_density(w2, wgrid, gamma)
    diff2 = (rho1 - rho2) ** 2
    # simple trapezoidal rule
    return float(np.trapz(diff2, wgrid)) ** 0.5

def ipsen_mikhailov_similarity(
    nodes_union: Set[frozenset],
    edges1: Set[Tuple[frozenset, frozenset]],
    edges2: Set[Tuple[frozenset, frozenset]],
    gamma: float = 0.08,
) -> Tuple[float, float]:
    """
    Returns (im_distance, im_similarity) on [0,1].
    - Distance computed on undirected versions over the *same* node set.
    - Similarity := 1 - d / d_max, where d_max is distance between empty and complete graph.
    """
    nodes = sorted(nodes_union, key=lambda x: (len(x), tuple(sorted(x))))
    # spectra
    l1 = _laplacian_eigs_undirected(nodes, edges1)
    l2 = _laplacian_eigs_undirected(nodes, edges2)
    d = _im_distance_from_spectra(l1, l2, gamma=gamma)

    # normalization baseline on same node set
    n = len(nodes)
    empty_edges: Set[Tuple[frozenset, frozenset]] = set()
    complete_edges: Set[Tuple[frozenset, frozenset]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            complete_edges.add((nodes[i], nodes[j]))
    L_empty = _laplacian_eigs_undirected(nodes, empty_edges)
    L_full  = _laplacian_eigs_undirected(nodes, complete_edges)

    dmax = _im_distance_from_spectra(L_empty, L_full, gamma=gamma)
    # guard
    if dmax <= 1e-12:
        sim = 1.0 if d <= 1e-12 else 0.0
    else:
        sim = max(0.0, min(1.0, 1.0 - d / dmax))
    return d, sim
# ----------------------------
# Tree structures and Newick
# ----------------------------

class TreeNode:
    def __init__(self, name: Optional[str] = None):
        self.name: Optional[str] = name
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None

    def is_leaf(self): return len(self.children) == 0
    def add_child(self, child: "TreeNode"):
        self.children.append(child); child.parent = self

    def __repr__(self):
        return f"Leaf({self.name})" if self.is_leaf() else f"Node({self.name}, k={len(self.children)})"
    
def parse_input_file_list(path: str) -> Tuple[List[str], List[str]]:
    """Reads a file where each line is 'tree_path\\tmeta_path'."""
    tree_paths = []
    meta_paths = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue # Skip empty or invalid lines
            try:
                tree_p, meta_p = line.split('\t', 1)
                tree_paths.append(tree_p.strip())
                meta_paths.append(meta_p.strip())
            except ValueError:
                print(f"Skipping malformed line: {line}")
    if not tree_paths:
         raise ValueError(f"No valid tree/meta path pairs found in {path}")
    # print(f"Read {len(tree_paths)} tree/meta pairs from {path}")
    return tree_paths, meta_paths

def iter_edges(root: TreeNode) -> Iterable[Tuple[TreeNode, TreeNode]]:
    """Yield (parent, child) for every directed edge in the rooted tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            yield (node, child)
            stack.append(child)


def count_edges(root: TreeNode) -> int:
    """Count number of directed edges in tree rooted at `root`."""
    return sum(1 for _ in iter_edges(root))


# -------------------------
# Union-only Fitch labeling
# -------------------------
#Modifed for handling TLS
def assign_union_potency(root: TreeNode, leaf_type_map: Dict[str, str]) -> Set[str]:
    """
    Post-order union-only labeling. Sets `node.potency` for every node (as a Python set).
    For leaves, looks up leaf_type_map[node.name] to get the leaf cell type.
    **If a leaf name from the tree is not found in the map (e.g., due to filtering),
    it assigns an empty potency set {}.**
    Returns the potency set at `root`.
    """
    # --- Check for internal node first ---
    if not root.is_leaf():
        union_set: Set[str] = set()
        for child in root.children:
            # Recursively call on children
            child_set = assign_union_potency(child, leaf_type_map)
            union_set |= child_set
        root.potency = union_set
        return root.potency

    # --- Handle Leaf Node ---
    else:
        if root.name is None:
            # Still raise error if leaf node lacks a name attribute entirely
            raise ValueError("Leaf node encountered with no name attribute.")

        # --- MODIFIED LOGIC: Use .get() for robustness ---
        leaf_type = leaf_type_map.get(root.name) # Returns None if not found

        if leaf_type is None:
            # Leaf name exists in tree but was not in the filtered map.
            # Assign an empty set, effectively ignoring this leaf for Fitch.
            root.potency = set()
            # Optional: Add a warning if you want to see which leaves are ignored
            # print(f"Warning: Leaf '{root.name}' not in map (likely filtered). Assigning empty potency.")
        else:
            # Leaf name was found in the map, assign its type as potency.
            root.potency = {leaf_type}
        # ------------------------------------------------

        return root.potency


# -------------------------
# Per-tree transition counts
# -------------------------
def per_tree_transition_counts(root: TreeNode) -> Counter:
    """
    Count transitions (parent_set -> child_set) for all direct edges in the tree,
    excluding edges where parent.potency == child.potency.
    Returns Counter with keys (frozenset_parent, frozenset_child) -> count (int).
    """
    C = Counter()
    for (u, v) in iter_edges(root):
        su = frozenset(u.potency if u.potency is not None else set())
        sv = frozenset(v.potency if v.potency is not None else set())
        if su != sv:
            C[(su, sv)] += 1
    return C


# -------------------------
# Aggregation + top-k picking
# -------------------------
def init_progenitors_union_fitch(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    k: int,
) -> Tuple[Dict[Tuple[frozenset, frozenset], float], Set[frozenset]]:
    """
    Run union-Fitch on each tree, compute normalized transition counts per tree (only real transitions),
    aggregate across trees, compute row-sums and return:
      - aggregated_transitions: dict ( (frozenset_i, frozenset_j) -> float )
      - Z_init: set of frozensets including:
          * ROOT potency (all leaf types)
          * all singleton potencies (each leaf type)
          * top (k-1) progenitor states (size >= 2, excluding ROOT)
    """
    if len(trees) != len(leaf_type_maps):
        raise ValueError("Provide exactly one leaf_type_map per tree (same order).")

    ROOT = frozenset(S)  # absolute root potency (all leaf types)
    aggregated_transitions: Dict[Tuple[frozenset, frozenset], float] = defaultdict(float)
    row_sum: Dict[frozenset, float] = defaultdict(float)

    for tree, ltm in zip(trees, leaf_type_maps):
        # Assign potencies
        assign_union_potency(tree, ltm)
        # Count only real transitions
        C_T = per_tree_transition_counts(tree)
        T = sum(C_T.values())  # number of actual transitions
        if T == 0:
            continue
        # Normalize and aggregate
        for (i_set, j_set), cnt in C_T.items():
            incr = cnt / T
            aggregated_transitions[(i_set, j_set)] += incr
            row_sum[i_set] += incr

    # Start Z_init with root and all singletons
    Z_init: Set[frozenset] = {ROOT} | {frozenset([cell]) for cell in S}

    # Candidates: only potency sets of size >= 2 (excluding ROOT)
    candidates = [ps for ps in row_sum.keys() if ps != ROOT and len(ps) >= 2]

    # Sort candidates by score: row_sum desc, then size desc, then lexicographic
    candidates.sort(key=lambda ps: (-row_sum[ps], -len(ps), tuple(sorted(ps))))

    # Take top (k-1) progenitors (excluding root, which is already in)
    top_progenitors = candidates[:max(0, k - 1)]

    # Add these progenitors to Z_init
    Z_init |= set(top_progenitors)

    return dict(aggregated_transitions), Z_init

def parse_newick(newick: str) -> TreeNode:
    # Helper: strip branch length and numeric-only labels
    def _clean_label(tok: str) -> str:
        # remove branch length: keep part before first ':'
        tok = tok.split(":", 1)[0].strip()
        # drop pure numeric internal labels like "357"
        if tok and tok.replace(".", "", 1).isdigit():
            return ""
        return tok

    s = newick.strip()
    if not s.endswith(";"): raise ValueError("Newick must end with ';'")
    s = s[:-1]; i = 0

    def parse() -> TreeNode:
        nonlocal i, s
        if i >= len(s): raise ValueError("Unexpected end")
        if s[i] == '(':
            i += 1
            node = TreeNode()
            while True:
                node.add_child(parse())
                if i >= len(s): raise ValueError("Unbalanced")
                if s[i] == ',':
                    i += 1; continue
                elif s[i] == ')':
                    i += 1; break
                else:
                    raise ValueError(f"Unexpected char: {s[i]} at {i}")

            # optional internal node label (may include branch length)
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if name:  # keep non-empty, non-numeric labels only
                node.name = name
            i = j
            return node
        else:
            # leaf label (may include branch length)
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if not name:
                raise ValueError("Leaf without name")
            i = j
            return TreeNode(name=name)

    root = parse()
    if i != len(s): raise ValueError(f"Trailing characters: '{s[i:]}'")
    return root

def to_newick(root: TreeNode) -> str:
    def rec(n: TreeNode) -> str:
        if n.is_leaf(): return n.name or ""
        return f"({','.join(rec(c) for c in n.children)}){n.name or ''}"
    return rec(root) + ";"


def read_newick_file(path: str) -> TreeNode:
    with open(path, "r") as f: s = f.read().strip()
    return parse_newick(s)

def write_newick_file(path: str, root: TreeNode):
    with open(path, "w") as f: f.write(to_newick(root) + "\n")

def random_tree_newick(n_leaves: int, leaf_prefix="L") -> Tuple[TreeNode, List[str]]:
    leaves = [TreeNode(f"{leaf_prefix}{i+1}") for i in range(n_leaves)]
    nodes = leaves[:]
    while len(nodes) > 1:
        k = 2 if len(nodes) < 4 else random.choice([2,2,2,3])
        k = min(k, len(nodes))
        picks = random.sample(nodes, k)
        for p in picks: nodes.remove(p)
        parent = TreeNode()
        for p in picks: parent.add_child(p)
        nodes.append(parent)
    return nodes[0], [l.name for l in leaves]

def collect_leaf_names(root: TreeNode) -> List[str]:
    out=[]
    def dfs(v):
        if v.is_leaf(): out.append(v.name)
        else:
            for c in v.children: dfs(c)
    dfs(root); return out

# ----------------------------
# Potency universe and structure
# ----------------------------

def all_nonempty_subsets(S: List[str], max_size: Optional[int]=None) -> List[FrozenSet[str]]:
    # Generate all non-empty subsets of the label universe S, optionally capped by max_size.
    # Each subset is returned as a frozenset so it can be used as a dict/set key elsewhere.
    # Used by: build_Z_active (to enumerate candidate potency sets).
    R=len(S); max_k = R if max_size is None else min(max_size, R)
    res=[]
    # k = subset size (from 1 up to max_k). We exclude k=0 (the empty set).
    for k in range(1, max_k+1):
        # itertools.combinations yields all size-k subsets of S (as tuples).
        # We wrap them in frozenset to make them hashable and order-invariant.
        for comb in itertools.combinations(S, k): res.append(frozenset(comb))
    return res

def singletons(S: List[str]) -> Set[FrozenSet[str]]:
    # Return the set of all singleton subsets {t} for each type t in S.
    # Singletons represent terminal/atomic potencies (always included in Z).
    # Used by: build_Z_active (baseline active nodes).
    return {frozenset([t]) for t in S}

def build_Z_active(S: List[str], fixed_k: Optional[int], max_potency_size: Optional[int], seed=0) -> Set[FrozenSet[str]]:
    # Construct the initial active potency set Z:
    # - Always include all singletons {t} for t in S.
    # - For multi-type potencies (|P| >= 2), either:
    #     * If fixed_k is not None: uniformly sample exactly fixed_k of them.
    #     * Else: include ALL multi-type potencies up to max_potency_size.
    # This "Z" forms the node set of the potency DAG used by Structure.
    # Used by: map_search (to initialize candidate structures).
    rng = random.Random(seed)
    P_all = all_nonempty_subsets(S, max_potency_size)   # all non-empty subsets up to size cap
    singles = singletons(S)                             # all {t}
    multis = [P for P in P_all if len(P)>=2]            # only multi-type potencies (size >= 2)
    Z = set(singles)                                    # start with all singletons
    if fixed_k is not None:
        if fixed_k > len(multis):
            raise ValueError("fixed_k too large")
        root = frozenset(S)
        Z.add(root)
        remaining_multis = [P for P in multis if P != root]
        Z.update(rng.sample(remaining_multis, fixed_k - 1))  # pick k-1 more
    else:
        Z.update(multis)
    return Z

def admissible_edge(P: FrozenSet[str], Q: FrozenSet[str], unit_drop: bool) -> bool:
    # Decide if an edge P -> Q is allowed in the potency DAG.
    # Constraints:
    #  - Q must be a proper subset of P (monotone decreasing).
    #  - If unit_drop=True, exactly one element must be dropped: |P \ Q| == 1.
    #  - No self-loops.
    # Used by: build_edges (to enumerate valid edges).
    if Q == P: return False
    if not Q.issubset(P): return False
    if len(Q) >= len(P): return False
    if unit_drop and len(P - Q) != 1: return False
    return True

def build_edges(Z_active: Set[FrozenSet[str]], forbid_fn=None, unit_drop=True) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
    # Build the adjacency dictionary A over the active potency set Z_active.
    # For every admissible pair (P, Q), create an edge indicator A[(P, Q)] = 1.
    # Optionally skip edges if forbid_fn(P,Q) returns True.
    # The unit_drop flag enforces |P \ Q| == 1 if True; otherwise any strict subset is allowed.
    # Used by: map_search (to initialize a connected structure so labels can "flow" down).
    A={}
    for P in Z_active:
        for Q in Z_active:
            if not admissible_edge(P,Q,unit_drop): continue
            if forbid_fn and forbid_fn(P,Q): continue
            A[(P,Q)] = 1
    return A

def build_mid_sized_connected_dag(Z_active, keep_prob=0.3, unit_drop=False, rng=None):
    """
    Build a valid mid-density DAG:
      • Uses only admissible edges
      • Guarantees connectivity from the root node (frozenset of all singletons)
      • Keeps density moderate, controlled by `keep_prob`
    """
    if rng is None:
        rng = random.Random()

    # --- Identify root node (the potency containing all singletons) ---
    root = frozenset().union(*Z_active)  # union of all labels gives the full set
    if root not in Z_active:
        raise ValueError("Root potency (all singletons) not present in Z_active.")

    nodes = list(Z_active)

    # --- Step 1: Build full admissible edge set ---
    full_edges = {
        (P, Q): 1
        for P in Z_active
        for Q in Z_active
        if P != Q and admissible_edge(P, Q, unit_drop)
    }

    # --- Step 2: Start with a spanning tree to guarantee connectivity ---
    A = {}
    visited = {root}
    to_visit = set(nodes) - {root}

    while to_visit:
        # pick a node already in the tree
        parent = rng.choice(list(visited))

        # find valid edges from parent to some unvisited node
        candidates = [(parent, q) for q in to_visit if (parent, q) in full_edges]

        if not candidates:
            # fallback: pick any edge between visited and unvisited nodes
            candidates = [
                (p, q) for p in visited for q in to_visit if (p, q) in full_edges
            ]

        edge = rng.choice(candidates)
        A[edge] = 1
        visited.add(edge[1])
        to_visit.remove(edge[1])

    # --- Step 3: Add extra edges randomly to reach desired density ---
    for edge in full_edges:
        if edge in A:
            continue
        if rng.random() < keep_prob:
            A[edge] = 1

    return A


def transitive_closure(labels: List[FrozenSet[str]], A: Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]) -> Dict[FrozenSet[str], Set[FrozenSet[str]]]:
    # Compute reachability (transitive closure) over the directed graph (labels, A).
    # Result: Reach[L] = set of nodes U such that there is a path L ->* U (including L itself).
    # Implementation details:
    #  - Build an index for labels and a boolean adjacency matrix M.
    #  - Set M[i][i] = True (reflexive reachability).
    #  - For each edge (P,Q) with A[(P,Q)]==1, mark M[i][j] = True.
    #  - Floyd–Warshall-style closure: if i->k and k->j then i->j.
    # Used by: Structure.__init__/recompute_reach (to query allowed label transitions during DP).
    idx = {L:i for i,L in enumerate(labels)}
    n=len(labels)
    M=[[False]*n for _ in range(n)]
    for i in range(n): M[i][i]=True                 # every node reaches itself
    for (P,Q),v in A.items():
        if v:
            i,j=idx[P],idx[Q]; M[i][j]=True         # direct edges from A

    # Triple loop closure (standard transitive closure).
    for k in range(n):
        Mk=M[k]
        for i in range(n):
            if M[i][k]:
                Mi=M[i]
                for j in range(n):
                    if Mk[j]: Mi[j]=True

    # Rehydrate into a dict keyed by the actual frozenset labels.
    Reach={L:set() for L in labels}
    for i,L in enumerate(labels):
        for j,U in enumerate(labels):
            if M[i][j]: Reach[L].add(U)
    return Reach


# ----------------------------
# DP over labelings (integrated Beta)
# ----------------------------

def compute_B_sets(root: TreeNode, leaf_to_type: Dict[str,str]) -> Dict[TreeNode, Set[str]]:
    # Build, for every node v in the tree, the set B[v] of observed types that appear
    # in v's subtree (union over leaves below v).
    #
    # Key detail for robustness (as you requested):
    # - If a leaf's name is missing from `leaf_to_type`, we *ignore* that leaf by contributing an empty set.
    #
    # Used by: score_structure() before DP, which passes B_sets into dp_tree_root_table().
    B={}
    def post(v: TreeNode) -> Set[str]:
        if v.is_leaf():
            # If the leaf has a mapping, add that single type; else contribute empty set.
            t = leaf_to_type.get(v.name)
            # Missing mapping? Ignore this leaf by contributing an empty set.
            B[v] = {t} if t is not None else set()
            return B[v]
        # Internal node: union of children's type-sets.
        acc=set()
        for c in v.children: acc |= post(c)
        B[v]=acc; return acc
    post(root); return B


def dp_tree_root_table(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
    prune_eps: float = 0.0
) -> Dict[Tuple[int,int],float]:
    
    label_index = {L: i for i, L in enumerate(active_labels)}
    memo: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
    def nid(v: TreeNode) -> int: return id(v)

    def M(v: TreeNode, P: Optional[FrozenSet[str]], depth: int) -> Dict[Tuple[int, int], float]:
        indent = "  " * depth
        key = (nid(v), -1 if P is None else label_index[P])
        if key in memo:
            return memo[key]

        # print(f"{indent}[DP] Node id={nid(v)}, Parent Label={P}")

        if v.is_leaf():
            # print(f"{indent}[DP] >> Leaf node. Returning base case log-table.")
            memo[key] = {(0, 0): 0.0} # log(1.0) = 0.0
            return memo[key]

        Bv = B_sets.get(v, set())
        # print(f"{indent}[DP] Observed types B(v): {Bv}")
        out = {}

        parent_reach = active_labels if P is None else list(Reach.get(P, []))
        # print(f"{indent}[DP] Node can take one of {len(parent_reach)} reachable labels.")

        for L in parent_reach:
            # print(f"{indent}  - Testing label L = {L}")
            if not Bv.issubset(L):
                # print(f"{indent}    -> Containment check FAILED.")
                continue
            
            # print(f"{indent}    -> Containment check PASSED.")
            o_local = len(L & Bv)
            d_local = len(L - Bv)
            # print(f"{indent}    -> Local counts (O,D) = ({o_local}, {d_local})")

            child_tabs = []
            ok = True
            for i, u in enumerate(v.children):
                # print(f"{indent}    -> Recursing on child {i+1}/{len(v.children)} (id={nid(u)})")
                tab = M(u, L, depth + 1)
                if not tab:
                    # print(f"{indent}    [ERROR] Child {i+1} returned an empty DP table. Invalid path.")
                    ok = False
                    break
                child_tabs.append(tab)
            if not ok: continue

            conv = child_tabs[0] if child_tabs else {(0, 0): 0.0} # log(1.0) = 0.0
            for t in child_tabs[1:]:
                conv = sparse_convolve_2d_log(conv, t, depth + 1)
            
            # print(f"{indent}    -> Convolved child log-tables have {len(conv)} entries.")
            for (Oc, Dc), log_w in conv.items():
                key_out = (Oc + o_local, Dc + d_local)
                current_log_w = out.get(key_out, -math.inf)
                out[key_out] = logsumexp(current_log_w, log_w)
        
        # print(f"{indent}[DP] >> DP log-table for node id={nid(v)} has {len(out)} entries.")
        memo[key] = out
        return memo[key]

    return M(root, None, 0)


def logsumexp(a, b):
    """Numerically stable log(exp(a) + exp(b))"""
    # This is a low-level utility; uncomment the print statement
    # if you need extreme detail, but it will be very verbose.
    # print(f"logsumexp({a:.2f}, {b:.2f})")
    if a == -math.inf: return b
    if b == -math.inf: return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def sparse_convolve_2d_log(A: Dict[Tuple[int,int],float], B: Dict[Tuple[int,int],float], depth: int) -> Dict[Tuple[int,int],float]:
    """ Convolution of sparse 2D tables where values are in log-space. """
    indent = "  " * depth
    # print(f"{indent}  [DEBUG log_convolve] Convolving tables of size {len(A)} and {len(B)}")
    if not A: return B.copy()
    if not B: return A.copy()
    
    out = {}
    for (o1, d1), log_w1 in A.items():
        for (o2, d2), log_w2 in B.items():
            key = (o1 + o2, d1 + d2)
            current_log_w = out.get(key, -math.inf)
            out[key] = logsumexp(current_log_w, log_w1 + log_w2)
            
    # print(f"{indent}  [DEBUG log_convolve] >> Resulting table has size {len(out)}")
    return out

def tree_marginal_from_root_table_log(C_log: Dict[Tuple[int,int],float]) -> float:
    """
    Calculates the final log marginal probability from a log-space DP table.
    Computes log( Sum[ exp(log_w + log_beta) ] )
    """
    # print(f"[DEBUG log_marginal] Calculating final log-likelihood from log-table with {len(C_log)} entries.")
    if not C_log:
        return -math.inf

    log_terms = []
    for (O, D), log_w in C_log.items():
        log_beta = math.lgamma(O+1) + math.lgamma(D+1) - math.lgamma(O+D+2)
        log_terms.append(log_w + log_beta)
    
    if not log_terms:
        # print("[DEBUG log_marginal] >> No valid terms found.")
        return -math.inf
        
    max_log = max(log_terms)
    if max_log == -math.inf:
        # print("[DEBUG log_marginal] >> Max log term is -inf.")
        return -math.inf

    sum_of_exp = sum(math.exp(lt - max_log) for lt in log_terms)
    final_logL = max_log + math.log(sum_of_exp)
    
    # print(f"[DEBUG log_marginal] >> Final logL = {final_logL:.4f} (max_log={max_log:.4f}, log(sum_exp)={math.log(sum_of_exp):.4f})")
    return final_logL


def tree_marginal_from_root_table(C: Dict[Tuple[int,int],float]) -> Tuple[float, float]:
    """
    Calculates the log of the marginal probability using a max-scaling trick
    to prevent numerical overflow.
    """
    # print(f"[DEBUG marginal] Calculating marginal from table with {len(C)} entries using max-scaling.")
    if not C:
        # print("[DEBUG marginal] >> Input table is empty. Returning -inf.")
        return -math.inf, -math.inf

    # Find the maximum weight in the DP table
    max_w = -math.inf
    for w in C.values():
        if w > max_w:
            max_w = w
    
    if max_w <= 0:
        # print(f"[DEBUG marginal] >> Max weight is not positive ({max_w}). Returning -inf.")
        return -math.inf, -math.inf

    log_max_w = math.log(max_w)
    # print(f"[DEBUG marginal] Max weight found: {max_w:.4e}, log_max_w: {log_max_w:.4f}")
    
    # Calculate the sum of probabilities, but scaled by the max weight
    scaled_sum = 0.0
    for (O, D), w in C.items():
        w_scaled = w / max_w
        scaled_sum += w_scaled * beta_integral(O, D)
    
    # print(f"[DEBUG marginal] Scaled sum of probabilities: {scaled_sum:.4e}")

    if scaled_sum <= 0:
        # print(f"[DEBUG marginal] >> Scaled sum is not positive ({scaled_sum}). Returning -inf.")
        return -math.inf, -math.inf

    log_scaled_sum = math.log(scaled_sum)
    # print(f"[DEBUG marginal] >> Returning (log_scaled_sum={log_scaled_sum:.4f}, log_max_w={log_max_w:.4f})")
    return log_scaled_sum, log_max_w


def beta_integral(O:int, D:int) -> float:
    # This function is usually numerically stable, but we can add a print
    # print(f"    [DEBUG beta] B({O+1}, {D+1})")
    return math.exp(math.lgamma(O+1) + math.lgamma(D+1) - math.lgamma(O+D+2))

def sparse_convolve_2d(A: Dict[Tuple[int,int],float], B: Dict[Tuple[int,int],float], depth: int) -> Dict[Tuple[int,int],float]:
    indent = "  " * depth
    # print(f"{indent}  [DEBUG convolve] Convolving tables of size {len(A)} and {len(B)}")
    if not A: return B.copy()
    if not B: return A.copy()
    out = defaultdict(float)
    for (o1, d1), w1 in A.items():
        for (o2, d2), w2 in B.items():
            out[(o1 + o2, d1 + d2)] += w1 * w2
    # print(f"{indent}  [DEBUG convolve] >> Resulting table has size {len(out)}")
    return dict(out)

# ----------------------------
# Priors and scoring
# ----------------------------

class Priors:
    def __init__(self,
                 potency_mode:str="fixed_k",  # "fixed_k" or "bernoulli"
                 fixed_k:int=2,
                 pi_P:float=0.25,    # used if potency_mode="bernoulli"
                 rho:float=0.25):    # edge Bernoulli prob
        # ------------------------------------------------------------
        # Stores hyperparameters for the prior over the structure F=(Z,A)
        #     Z: The latent assignment of “potencies” or features to nodes (the sets like {A,B,C}, {B,C,D}, etc. that you saw in the MAP output).
        #     A: The active structure (the adjacency or edge set) consistent with those potencies — basically the graph/hypergraph that the algorithm thinks best explains the observed trees.
        #   - potency_mode: which prior to use over active multi-type potencies Z
        #       * "fixed_k": exactly k multi-type potencies are active (uniform over choices)
        #       * "bernoulli": each multi-type potency is independently active with prob pi_P
        #   - fixed_k: number of multi-type potencies when potency_mode == "fixed_k"
        #   - pi_P: inclusion probability for each multi-type potency when using "bernoulli" mode
        #   - rho: prior probability that any admissible edge (P->Q) exists
        # ------------------------------------------------------------
        self.potency_mode=potency_mode
        self.fixed_k=fixed_k
        self.pi_P=pi_P
        self.rho=rho

    def log_prior_Z(self, S: List[str], Z_active:Set[FrozenSet[str]])->float: #Z_active = the set of active potencies (both singletons and multis).
        # ------------------------------------------------------------
        # Computes log P(Z): the log prior over WHICH potencies are active.
        #
        # Inputs:
        #   - S: list of all cell types (leaf types), e.g., ["A","B","C","D"]
        #   - Z_active: set of active potencies (as frozensets). Includes singletons by construction.
        #
        # Key ideas:
        #   - Singletons are always considered active (terminal states), we don't penalize/score them.
        #   - We only place a prior over multi-type potencies (size >= 2).
        #   - Two modes:
        #       * "fixed_k": valid only if exactly `fixed_k` multis are active.
        #                    Prior is uniform over all C(M, k) choices, where M = #all possible multis.
        #       * "bernoulli": each possible multi is included independently with prob pi_P.
        #                      log prior sums log(pi_P) for included multis and log(1-pi_P) for excluded ones.
        # Returns:
        #   - log prior (float), or -inf if configuration violates "fixed_k".
        # ------------------------------------------------------------
        singles = singletons(S)
        multis = [P for P in Z_active if len(P)>=2] #P is a particular potency set
        # count available multi potencies (for fixed-k uniform)
        all_multis = [P for P in all_nonempty_subsets(S) if len(P)>=2]

        if self.potency_mode=="fixed_k":
            # ------------------------------
            # Uniform prior over all subsets of multi-type potencies with EXACTLY k elements.
            # If the current Z_active has not exactly k multis, return -inf (outside prior support).
            # Otherwise, log prior = -log( number of such subsets ) = -log( nCk ).
            # ------------------------------
            k=len(multis)
            if k!=self.fixed_k:
                return float("-inf")
            # uniform over all C(|all_multis|, k)
            total = math.comb(len(all_multis), k) #this is nCk
            return -math.log(total) if total>0 else float("-inf")
        else:
            # ------------------------------
            # Bernoulli prior on each multi-type potency:
            #   P(Z) = ∏_{P in all_multis} pi_P^{I[P in Z]} (1 - pi_P)^{I[P not in Z]}
            # We sum logs across all possible multi-type potencies (singletons ignored).
            # ------------------------------
            k_log=0.0
            for P in all_multis:
                if P in Z_active: k_log += math.log(self.pi_P)
                else: k_log += math.log(1-self.pi_P)
            return k_log

    def log_prior_A(self, Z_active:Set[FrozenSet[str]], A:Dict[Tuple[FrozenSet[str],FrozenSet[str]],int], unit_drop=True)->float:
        # ------------------------------------------------------------
        # Computes log P(A | Z): the log prior over EDGE EXISTENCE between active potencies.
        #
        # Inputs:
        #   - Z_active: set of active potencies (nodes in the potency DAG)
        #   - A: adjacency dictionary mapping (P,Q) -> {0,1}, indicating whether edge P->Q is present
        #   - unit_drop: if True, an admissible edge must drop EXACTLY one fate (|P\Q| == 1);
        #                otherwise any monotone subset drop (Q ⊂ P) is admissible.
        #
        # Prior:
        #   - For every admissible pair (P,Q):
        #         A_{P->Q} ~ Bernoulli(rho)
        #     So:
        #         log P(A|Z) = ∑_{(P,Q) admissible} [ A_{P->Q} log(rho) + (1 - A_{P->Q}) log(1 - rho) ]
        #
        # Notes:
        #   - "Admissible" enforces graph shape constraints (subset-monotone and possibly unit-drop).
        #   - If an edge (P,Q) is not admissible, it does not contribute to the product/sum at all.
        # ------------------------------------------------------------
        labels=list(Z_active)
        # admissible set is pairs with subset monotone (and optionally unit-drop)
        logp=0.0
        for P in labels:
            for Q in labels:
                if admissible_edge(P,Q,unit_drop):
                    # a == 1 if the edge is present in A, else 0
                    a = 1 if A.get((P,Q),0)==1 else 0
                    # add Bernoulli log-prob for this edge
                    logp += math.log(self.rho) if a==1 else math.log(1-self.rho)
        return logp

# ----------------------------
# Structure container and proposals
# ----------------------------

class Structure:
    def __init__(self,
                 S: List[str],
                 Z_active: Set[FrozenSet[str]],
                 A: Dict[Tuple[FrozenSet[str],FrozenSet[str]],int],
                 unit_drop: bool = True):
        # The "model structure" F = (Z, A) that the search optimizes.
        # - S: universe of primitive types (e.g., {"-7","-8","-9"}).
        # - Z_active: active potency sets (nodes of the DAG). Always includes singletons {t} for t in S.
        #             May also include multi-type sets like {"-7","-8"} depending on the prior/moves.
        # - A: adjacency over Z_active, A[(P,Q)] ∈ {0,1}, indicating presence of edge P -> Q.
        #      Edges are subset-monotone (and may enforce |P\Q|=1 if unit_drop=True).
        # - unit_drop: if True, only allow edges that drop exactly one element (|P\Q| == 1).
        #
        # Where it’s used:
        # - Created/updated inside map_search() during the annealed hill climb.
        # - Passed into score_structure() which uses:
        #     * struct.labels_list (sorted Z) and
        #     * struct.Reach (transitive closure over A)
        #   to run the DP (dp_tree_root_table) and compute the likelihood.
        self.S=S
        self.Z_active=set(Z_active)  # copy to decouple from caller; Z includes singletons and selected multis
        self.A=dict(A)               # copy adjacency dict (edges)
        self.unit_drop=unit_drop
        # A consistent ordering of labels (frozensets) for indexing/memoization in DP
        self.labels_list=self._sorted_labels()
        # Reachability closure used by DP to constrain child labels given a parent label
        self.Reach = transitive_closure(self.labels_list, self.A)

    def _sorted_labels(self)->List[FrozenSet[str]]:
        # Provide a stable, human-logical ordering of the active labels:
        #   1) by set size (|L|), then
        #   2) lexicographically by the sorted elements of the set.
        # This keeps DP indices stable and makes printed output neat.
        return sorted(list(self.Z_active), key=lambda x: (len(x), tuple(sorted(list(x)))))

    def recompute_reach(self):
        # Recompute both the sorted label list and the transitive closure Reach
        # after any structural change (adding/removing potencies or edges).
        # Called by all propose_* methods after they mutate Z_active or A.
        self.labels_list=self._sorted_labels()
        self.Reach = transitive_closure(self.labels_list, self.A)

    def clone(self)->"Structure":
        # Return a deep-enough copy to test/accept a proposal without mutating the current state.
        # Used heavily during the stochastic search (map_search) to try local moves.
        return Structure(self.S, set(self.Z_active), dict(self.A), self.unit_drop)

    # --- Moves ---
    def potencies_multi_all(self)->List[FrozenSet[str]]:
        # Enumerate ALL candidate multi-type potencies (|P|>=2) over S.
        # This is the proposal pool for adding or swapping potencies.
        return [P for P in all_nonempty_subsets(self.S) if len(P)>=2]

    def propose_add_potency(self, rng:random.Random)->Optional["Structure"]:
        # Propose: add one multi-type potency (node) not yet in Z_active.
        # Leaves edges A unchanged (edge proposals are separate); only Z is changed here.
        # Returns a NEW Structure if a candidate exists; otherwise None.
        candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        new.Z_active.add(P)
        # Keep the edge set as-is; we only ensure Reach is recomputed to stay consistent.
        new.recompute_reach()
        return new

    def propose_remove_potency(self, rng:random.Random)->Optional["Structure"]:
        # Propose: remove one existing multi-type potency from Z_active.
        # Also removes any incident edges from A that reference that potency.
        # Returns a NEW Structure, or None if there are no multis to remove.
        candidates = [P for P in self.Z_active if len(P)>=2]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        # remove potency and incident edges
        new.Z_active.remove(P)
        new.A = {e:v for e,v in new.A.items() if P not in e}
        new.recompute_reach()
        return new

    def propose_swap_potency(self, rng:random.Random)->Optional["Structure"]:
        # Propose: swap out one existing multi-type potency for a different one not currently active.
        # Useful in fixed-k mode to keep the number of multi-type nodes constant while exploring.
        remove_candidates = [P for P in self.Z_active if len(P)>=2]
        add_candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not remove_candidates or not add_candidates: return None
        P_rm = rng.choice(remove_candidates)
        P_add = rng.choice(add_candidates)
        print(f"P_rm : {P_rm} and P_add : {P_add}")
        new = self.clone()
        new.Z_active.remove(P_rm)
        new.A = {e:v for e,v in new.A.items() if P_rm not in e}
        new.Z_active.add(P_add)
        new.recompute_reach()
        return new

    def all_edge_pairs(self)->List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        # Enumerate all admissible ordered pairs (P,Q) among the currently active potencies.
        # Uses admissible_edge(P,Q, unit_drop) to enforce subset-monotone (and unit-drop if requested).
        # This is the proposal pool for add-edge moves.
        L=list(self.Z_active)
        pairs=[]
        for P in L:
            for Q in L:
                if admissible_edge(P,Q,self.unit_drop):
                    pairs.append((P,Q))
        return pairs

    def propose_add_edge(self, rng:random.Random)->Optional["Structure"]:
        # Propose: add a single admissible edge (P->Q) that is currently absent (A[(P,Q)] == 0).
        # Returns a NEW Structure or None if no addable edge exists.
        pairs = [e for e in self.all_edge_pairs() if self.A.get(e,0)==0]
        if not pairs: return None
        e = rng.choice(pairs)
        new = self.clone()
        new.A[e]=1
        new.recompute_reach()
        return new

    def propose_remove_edge(self, rng:random.Random)->Optional["Structure"]:
        # Propose: remove a single existing edge (P->Q) where A[(P,Q)] == 1.
        # Returns a NEW Structure or None if there are no edges to remove.
        edges = [e for e,v in self.A.items() if v==1]
        if not edges: return None
        e = rng.choice(edges)
        new = self.clone()
        del new.A[e]
        new.recompute_reach()
        return new


def _worker_viterbi_flow(args):
    """
    Worker to calculate flow contributions from a single tree.
    Recomputes B_sets locally to ensure object identity match with the pickled tree copy.
    """
    tree, leaf_map, labels_list, Reach, leaf_counts = args
    
    # Recompute B_sets locally so keys match the 'tree' object in this process
    B_sets = compute_B_sets(tree, leaf_map)
    
    # Run Viterbi DP
    root_table, memo = dp_tree_root_viterbi(
        tree, labels_list, Reach, B_sets
    )
    
    # Reconstruct MAP labeling
    L_MAP = find_best_viterbi_labeling(tree, root_table, memo)
    
    local_flow = defaultdict(float)
    
    if not L_MAP:
        return local_flow
    
    # Aggregate flow for this tree
    for (v, u) in iter_edges(tree):
        P = L_MAP.get(v)
        Q = L_MAP.get(u)
        
        if P is None or Q is None:
            continue
            
        if P != Q:
            # Weighted flow based on child's leaf descendants
            flow_weight = leaf_counts.get(u, 1)
            local_flow[(P, Q)] += flow_weight
            
    return local_flow

def sparse_convolve_2d_viterbi(
    A: Dict[Tuple[int, int], Tuple[float, Any]], 
    B: Dict[Tuple[int, int], Tuple[float, Any]],
    depth: int
) -> Dict[Tuple[int, int], Tuple[float, Any]]:
    """
    Viterbi-style convolution for (O,D) tables.
    
    Input tables A and B map:
      (O, D) -> (max_log_prob, backpointer_for_this_OD_sum)
      
    Returns a new table in the same format.
    """
    indent = "  " * depth
    if not A: return B
    if not B: return A
    
    out: Dict[Tuple[int, int], Tuple[float, Any]] = {}
    
    for (o1, d1), (log_w1, bp1) in A.items():
        for (o2, d2), (log_w2, bp2) in B.items():
            key = (o1 + o2, d1 + d2)
            new_log_w = log_w1 + log_w2
            
            current_log_w, _ = out.get(key, (-math.inf, None))
            
            # Viterbi step: check if this new path is better
            if new_log_w > current_log_w:
                # This path is the new best one for (o1+o2, d1+d2)
                # We store the *combined* backpointer
                
                ### <--- THIS IS THE FIX ###
                # We concatenate the backpointer tuples, not nest them.
                out[key] = (new_log_w, bp1 + bp2)
                
    return out
    
def dp_tree_root_viterbi(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
) -> Tuple[Dict[Tuple[int, int], Tuple[float, Any]], Dict[TreeNode, Any]]:
    """
    Performs the Viterbi DP to find the single best labeling.
    This modifies the original `dp_tree_root_table`.
    
    Returns:
     1. root_table: The final (O,D) -> (max_log_prob, backpointer) table for the root.
     2. memo: A full memoization table for all nodes, used by the backtracking function.
    """
    label_index = {L: i for i, L in enumerate(active_labels)}
    
    # memo[v] = Dict[Label L, (scores, backpointers)]
    #   scores = Dict[(O_subtree, D_subtree), max_log_prob]
    #   backpointers = Dict[(O_subtree, D_subtree), child_backpointer_map]
    #     child_backpointer_map = Dict[child_node, (Child_Label_Q, (O_child, D_child))]
    
    memo: Dict[TreeNode, Dict[FrozenSet[str], Tuple[
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int], Dict[TreeNode, Tuple[FrozenSet[str], Tuple[int, int]]]]
    ]]] = {}

    def M(v: TreeNode, P_parent: Optional[FrozenSet[str]], depth: int):
        
        # Determine allowed labels for this node 'v'
        if P_parent is None:
            allowed_labels = list(Reach.keys()) # Root can be anything
        else:
            allowed_labels = list(Reach.get(P_parent, set()))

        if v in memo:
            # We already computed the full table for this node
            return memo[v]

        indent = "  " * depth
        
        if v.is_leaf():
            # --- Leaf Node Base Case ---
            node_results: Dict[FrozenSet[str], Tuple[
                Dict[Tuple[int, int], float],
                Dict[Tuple[int, int], Dict[TreeNode, Tuple[FrozenSet[str], Tuple[int, int]]]]
            ]] = {}
            
            for L in allowed_labels:
                if not B_sets[v].issubset(L):
                    # This label is not allowed
                    continue
                
                o_local = len(L & B_sets[v])
                d_local = len(L - B_sets[v])
                
                # The score is log(1.0) = 0. We apply Beta func at the end.
                # The "backpointer" is empty.
                scores = {(o_local, d_local): 0.0}
                backpointers = {(o_local, d_local): {}}
                node_results[L] = (scores, backpointers)

            memo[v] = node_results
            return node_results

        # --- Internal Node Recursive Case ---
        Bv = B_sets.get(v, set())
        node_results = {}

        for L in allowed_labels:
            if not Bv.issubset(L):
                continue
            
            o_local = len(L & B_sets[v])
            d_local = len(L - B_sets[v])

            # Get tables from all children, assuming 'v' is labeled 'L'
            child_label_results = [M(u, L, depth + 1) for u in v.children]

            # We need to find the *best* combination of child labels
            # This is complex. Let's simplify.
            
            # --- Viterbi Convolution of Children ---
            # We need to convolve the results from all children
            
            # For each child, find its *best* label Q and the (O,D) table
            # from that best Q. This is still wrong.
            
            # Let's retry the convolve logic.
            
            child_tables = []
            all_children_valid = True
            for child_node, child_result_map in zip(v.children, child_label_results):
                # child_result_map = Dict[Child_Label_Q, (scores, backpointers)]
                
                # Combine all tables from all possible labels Q for this child
                child_combined_scores: Dict[Tuple[int, int], Tuple[float, Any]] = {}
                
                for Q, (scores, backpointers) in child_result_map.items():
                    for od_key, log_prob in scores.items():
                        # We store a backpointer to (Q, od_key)
                        bp = (Q, od_key) 
                        current_log_prob, _ = child_combined_scores.get(od_key, (-math.inf, None))
                        
                        if log_prob > current_log_prob:
                            child_combined_scores[od_key] = (log_prob, bp)
                
                if not child_combined_scores:
                    all_children_valid = False
                    break
                
                child_tables.append((child_node, child_combined_scores))
            
            if not all_children_valid:
                continue

            # Convolve the child tables
            # {(O,D): (log_prob, ((bp_c1), (bp_c2), ...))}
            conv_scores: Dict[Tuple[int, int], Tuple[float, Any]] = { (0,0): (0.0, tuple()) }
            if child_tables:
                conv_scores = child_tables[0][1]
                # Store node in backpointer
                conv_scores = {od: (lp, ((child_tables[0][0], bp),)) for od, (lp, bp) in conv_scores.items()}
                
                for child_node, child_table in child_tables[1:]:
                    # Wrap child_table backpointers
                    wrapped_child_table = {od: (lp, ((child_node, bp),)) for od, (lp, bp) in child_table.items()}
                    conv_scores = sparse_convolve_2d_viterbi(conv_scores, wrapped_child_table, depth + 1)

            # Now, add our local (o,d) and store in node_results[L]
            final_scores = {}
            final_backpointers = {}
            for (Oc, Dc), (log_w, bp_tuple) in conv_scores.items():
                key_out = (Oc + o_local, Dc + d_local)
                final_scores[key_out] = log_w
                
                # Convert tuple of ((node, (Q, od)), ...) to a dict
                bp_map = {node: bp for node, bp in bp_tuple}
                final_backpointers[key_out] = bp_map

            node_results[L] = (final_scores, final_backpointers)

        memo[v] = node_results
        return node_results

    # --- Main call ---
    # We pass P_parent=None for the root, which allows all labels
    root_result_map = M(root, None, 0)
    
    # Combine tables from all possible root labels
    final_root_table: Dict[Tuple[int, int], Tuple[float, Any]] = {}
    for L_root, (scores, backpointers) in root_result_map.items():
        for od_key, log_prob in scores.items():
            # Backpointer for the root is its label (L_root) and its (O,D) key
            bp = (L_root, od_key)
            current_log_prob, _ = final_root_table.get(od_key, (-math.inf, None))
            
            if log_prob > current_log_prob:
                final_root_table[od_key] = (log_prob, bp)
                
    return final_root_table, memo


def find_best_viterbi_labeling(
    root: TreeNode,
    root_table: Dict[Tuple[int, int], Tuple[float, Any]],
    memo: Dict[TreeNode, Any]
) -> Dict[TreeNode, FrozenSet[str]]:
    """
    Backtracks through the Viterbi memoization tables to reconstruct
    the single most probable labeling, L_MAP.
    """
    
    # 1. Find the best (O,D) at the root
    best_score = -math.inf
    best_OD = None
    best_root_bp = None
    
    for (O, D), (log_w, bp) in root_table.items():
        log_beta = math.lgamma(O+1) + math.lgamma(D+1) - math.lgamma(O+D+2)
        score = log_w + log_beta
        if score > best_score:
            best_score = score
            best_OD = (O, D)
            best_root_bp = bp

    if best_root_bp is None:
        # This tree is impossible to label
        return {}
        
    # 2. Start the recursive backtracking
    labeling: Dict[TreeNode, FrozenSet[str]] = {}
    
    def reconstruct(v: TreeNode, L_v: FrozenSet[str], OD_v: Tuple[int, int]):
        """Recursively reconstructs the labeling."""
        
        # Set the label for this node
        labeling[v] = L_v
        
        if v.is_leaf():
            return # Reached the end
            
        # Get the backpointer map for this state
        try:
            # v_results = memo[v]
            # L_v_results = v_results[L_v]
            # L_v_backpointers = L_v_results[1]
            # child_bp_map = L_v_backpointers[OD_v]
            child_bp_map = memo[v][L_v][1][OD_v]
        except KeyError:
            # This should not happen if the tables are built correctly
            print(f"KeyError during backtracking: v={v}, L_v={L_v}, OD_v={OD_v}")
            return

        # Recurse for all children
        for u in v.children:
            if u not in child_bp_map:
                # This child had no valid labels, which is a problem
                # But we'll continue, as it might just be an empty/filtered child
                continue
                
            (L_u, OD_u) = child_bp_map[u]
            reconstruct(u, L_u, OD_u)

    # 3. Start the recursion from the root
    (L_root, OD_root) = best_root_bp
    if L_root is not None:
        reconstruct(root, L_root, OD_root)
        
    return labeling

def calculate_viterbi_flow(
    trees: List[TreeNode],
    F_full: Structure, 
    all_B_sets: List[Dict[TreeNode, Set[str]]], # Kept for signature compatibility, but unused in parallel workers
    leaf_type_maps: List[Dict[str, str]],
    num_cores: int = 1  # <--- Added Argument
) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], float]:
    """
    Calculates the "Viterbi flow" w(P,Q) for all potential edges using parallel workers.
    """
    print(f"Calculating Viterbi flow w(P,Q) for all trees (cores={num_cores})...")
    viterbi_flow = defaultdict(float)
    
    # 1. Pre-calculate leaf counts for all nodes (Fast, serial)
    leaf_counts_all_trees = []
    for tree in trees:
        counts = {}
        def post_order_leaf_count(v):
            if v.is_leaf():
                counts[v] = 1
                return 1
            count = sum(post_order_leaf_count(u) for u in v.children)
            counts[v] = count
            return count
        post_order_leaf_count(tree)
        leaf_counts_all_trees.append(counts)

    # 2. Prepare Parallel Tasks
    # We pass leaf_type_maps instead of B_sets to ensure workers can recompute B_sets safely
    tasks = []
    for tree, leaf_map, counts in zip(trees, leaf_type_maps, leaf_counts_all_trees):
        tasks.append((tree, leaf_map, F_full.labels_list, F_full.Reach, counts))

    # 3. Execute
    if num_cores > 1:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Map returns an iterator, convert to list to force execution
            results = list(executor.map(_worker_viterbi_flow, tasks))
            
            # Aggregation (Must be done on main thread)
            for local_flow in results:
                for edge, weight in local_flow.items():
                    viterbi_flow[edge] += weight
    else:
        # Serial fallback
        for task in tasks:
            local_flow = _worker_viterbi_flow(task)
            for edge, weight in local_flow.items():
                viterbi_flow[edge] += weight
                
    return viterbi_flow
        
# ----------------------------
# Scoring: log posterior
# ----------------------------
# --- Helper for Beta-Binomial Log-Likelihood ---
def calc_log_beta_score(L: FrozenSet[str], Bv: Set[str]) -> float:
    """
    Calculates log P(Data | Label) for a single node.
    P ~ Beta(1,1) integrated out -> 1 / ((N+1) * Binom(N, O))
    Simplifies to log-beta function.
    """
    O = len(L & Bv)
    D = len(L - Bv)
    # log Beta(O+1, D+1) = lgamma(O+1) + lgamma(D+1) - lgamma(O+D+2)
    return math.lgamma(O + 1) + math.lgamma(D + 1) - math.lgamma(O + D + 2)

def greedy_tree_log_likelihood(
    root: TreeNode, 
    active_labels: List[FrozenSet[str]], 
    B_sets: Dict[TreeNode, Set[str]]
) -> float:
    """
    Calculates tree likelihood using a GREEDY BOTTOM-UP approach (Beam Size = 1).
    
    Logic:
      1. For leaves: Pick the single label L that maximizes local score. Lock it in.
      2. For internals: Must pick an L that is a superset of all locked-in children.
         Among valid Ls, pick the one maximizing local score. Lock it in.
    
    Returns:
      - Total log-likelihood of the tree (or -inf if greedy path breaks connectivity).
    """
    # Increase recursion limit for deep trees
    sys.setrecursionlimit(5000)

    # Cache for the single chosen label for each node
    # node_id -> chosen_label
    chosen_labels = {}

    def solve(v: TreeNode) -> float:
        Bv = B_sets.get(v, set())

        # --- 1. Process Children First (Bottom-Up) ---
        children_score_sum = 0.0
        child_constraints = set() # The union of all children's chosen potencies

        for child in v.children:
            s_child = solve(child)
            if s_child == -math.inf:
                return -math.inf # Propagate failure
            
            children_score_sum += s_child
            
            # Retrieve the child's locked decision
            child_label = chosen_labels[id(child)]
            child_constraints.update(child_label)

        # --- 2. Greedy Decision for Current Node v ---
        best_L = None
        best_local_score = -math.inf

        # We must choose an L that:
        # a) Contains all observed data at v (Bv)
        # b) Is a superset of the union of all children's chosen labels (child_constraints)
        
        # Optimization: Pre-check if child constraints are impossible (e.g. asking for types not in S)
        # (Skipped here, assuming Z contains the root)

        for L in active_labels:
            # Check constraint (a)
            if not Bv.issubset(L):
                continue
            
            # Check constraint (b) - The Greedy Constraint
            if not L.issuperset(child_constraints):
                continue

            # If valid, calculate local score
            score = calc_log_beta_score(L, Bv)
            
            # Greedy: Maximize local score
            if score > best_local_score:
                best_local_score = score
                best_L = L

        # --- 3. Lock It In or Fail ---
        if best_L is None:
            # Dead End: No label exists in Z that covers the greedy choices of children
            return -math.inf
        
        chosen_labels[id(v)] = best_L
        
        # Total score for this subtree = Local Score + Sum of Children's Total Scores
        return best_local_score + children_score_sum

    return solve(root)

def _worker_score_single_tree(args: tuple) -> float:
    """
    Worker function to score a single tree in parallel.
    Args: (tree, leaf_map, struct_S, struct_labels, struct_Reach, prune_eps)
    """
    # Unpack arguments
    # Note: We pass specific fields of Structure to ensure clean pickling
    # and avoid passing the whole object if unnecessary, though passing the object is also fine.
    tree, leaf_map, struct_labels, struct_Reach, prune_eps = args
    
    # Safety: Deep trees can hit recursion limits in new processes
    sys.setrecursionlimit(5000) 
    
    # 1. Compute B_sets locally (cheap)
    B_sets = compute_B_sets(tree, leaf_map)
    root_labels = B_sets.get(tree, set())

    if not root_labels:
        return 0.0

    # 2. Run DP
    C_log = dp_tree_root_table(tree, struct_labels, struct_Reach, B_sets, prune_eps=prune_eps)

    if not C_log:
        return float("-inf")

    # 3. Compute Marginal
    tree_logL = tree_marginal_from_root_table_log(C_log)
    
    if not math.isfinite(tree_logL):
        return float("-inf")
        
    return tree_logL

def score_structure_parallel(
    struct: Structure,
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
    priors: Priors,
    num_cores: int = 1,
    prune_eps: float = 0.0
) -> Tuple[float, List[float]]:
    """
    Parallelized version of score_structure for final reporting.
    """
    # 1. Calculate Priors
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp):
        return float("-inf"), []
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    # 2. Prepare Tasks 
    # Tuple format: (tree, leaf_map, labels, Reach, prune_eps)
    tasks = [
        (t, lm, struct.labels_list, struct.Reach, prune_eps)
        for t, lm in zip(trees, leaf_type_maps)
    ]

    # 3. Execute
    if num_cores > 1:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Reuse the worker from the previous step
            results = list(executor.map(_worker_score_single_tree, tasks))
    else:
        results = [_worker_score_single_tree(t) for t in tasks]

    # 4. Sum Results
    logLs = []
    for r in results:
        if not math.isfinite(r):
            logLs.append(float("-inf"))
        else:
            logLs.append(r + logp) # Add prior to every tree

    return sum(logLs), logLs

def score_structure(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    priors: Priors,
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # Compute the (log) posterior score of a candidate structure F = (Z_active, A).

    # ---- Prior over structure F ----
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp):
        return float("-inf"), []
    # Add the edge prior, which is the key difference in this function
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    # ---- Likelihood over all trees ----
    logLs = []
    for root, leaf_to_type in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_to_type)
        root_labels = B_sets.get(root, set())

        if not root_labels:
            logLs.append(0.0)
            continue

        # CORRECTED SECTION: Use the log-space DP and marginalization functions
        # 1. Call the DP function that returns a log-space table.
        C_log = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)

        if not C_log:
            # If the DP table is empty, this structure is impossible for this tree.
            return float("-inf"), []

        # 2. Call the marginalization function that works entirely in log-space.
        tree_logL = tree_marginal_from_root_table_log(C_log)

        if not math.isfinite(tree_logL):
            # If the final log-likelihood is not a valid number, abort.
            return float("-inf"), []

        # 3. Append the per-tree log-likelihood directly. No need for math.log().
        logLs.append(tree_logL + logp)

    # Total posterior score = log prior + sum of per-tree log-likelihoods
    total_log_post = sum(logLs)
    return total_log_post, logLs

def score_structure_greedy_likelihood(
    struct: Structure,
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    priors: Priors
) -> float:
    """
    Fast scoring wrapper using the Greedy strategy.
    """

    total_logL = 0.0

    # 2. Loop over trees
    for root, leaf_map in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_map)
        
        # Call the greedy solver
        tree_score = greedy_tree_log_likelihood(
            root, 
            struct.labels_list, 
            B_sets
        )
        
        if tree_score == -math.inf:
            return float("-inf")
            
        total_logL += (tree_score)

    return total_logL

def score_structure_greedy(
    struct: Structure,
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    priors: Priors
) -> float:
    """
    Fast scoring wrapper using the Greedy strategy.
    """
    # 1. Priors (Z and A)
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp): return float("-inf")
    
    # Note: In Z-only search, A is often implicit/ignored, but if present, we score it.
    # If unit_drop_edges=False (fully connected), log_prior_A might be constant or 0.
    # We add it here for consistency.
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    total_logL = 0.0

    # 2. Loop over trees
    for root, leaf_map in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_map)
        
        # Call the greedy solver
        tree_score = greedy_tree_log_likelihood(
            root, 
            struct.labels_list, 
            B_sets
        )
        
        if tree_score == -math.inf:
            return float("-inf")
            
        total_logL += (tree_score+logp)

    return total_logL

def score_structure_no_edge_prior(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    priors: Priors,
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # print("\n" + "="*50)
    # print("=== STARTING SCORE_STRUCTURE ===")
    
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    # print(f"[DEBUG score_structure] Log Prior P(Z): {logp}")
    if not math.isfinite(logp):
        # print("[DEBUG score_structure] P(Z) is -inf. ABORTING.")
        return float("-inf"), []

    logLs = []
    for i, (root, leaf_to_type) in enumerate(zip(trees, leaf_type_maps)):
        # print(f"\n--- Processing Tree {i+1}/{len(trees)} ---")
        B_sets = compute_B_sets(root, leaf_to_type)
        root_labels = B_sets.get(root, set())
        # print(f"[DEBUG score_structure] Tree {i+1} has {len(root_labels)} unique types in its leaves.")

        if not root_labels:
            # print(f"[DEBUG score_structure] Tree {i+1} has no mapped leaves. LogL = 0.0")
            logLs.append(0.0)
            continue
        
        C_log = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)
        if not C_log:
            return float("-inf"), []

        # Calculate the final log-likelihood directly
        tree_logL = tree_marginal_from_root_table_log(C_log)
        
        if not math.isfinite(tree_logL):
            return float("-inf"), []
        
        logLs.append(tree_logL + logp)

    # print(f"=== FINISHED SCORE_STRUCTURE | Total Log Posterior: {sum(logLs)} ===")
    # print("="*50 + "\n")
    return sum(logLs), logLs

def get_log_likelihood(
    struct: Structure,
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    prune_eps: float = 0.0
) -> float:
    """
    Calculates ONLY the log-likelihood part of the score: Sum[ log P(T|F) ].
    This is the 'L' term needed for AIC/BIC.
    """
    total_logL = 0.0
    for root, leaf_to_type, B_sets in zip(trees, leaf_type_maps, all_B_sets):
        # B_sets = compute_B_sets(root, leaf_to_type) # Already precomputed
        root_labels = B_sets.get(root, set())
        
        if not root_labels:
            # This tree has no mapped leaves, logL = log(1) = 0
            continue 

        C_log = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)

        if not C_log:
            # If the DP table is empty, this structure is impossible for this tree.
            return -math.inf 

        tree_logL = tree_marginal_from_root_table_log(C_log)
        
        if not math.isfinite(tree_logL):
            # If the final log-likelihood is not a valid number, abort.
            return -math.inf 
        
        total_logL += tree_logL
    
    return total_logL


# deterministically build ALL admissible edges for a given Z (keeps MCMC symmetric/easy)
def _full_edges_for_Z(Z_active: Set[FrozenSet[str]], unit_drop_edges: bool) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
    A = {}
    Zl = list(Z_active)
    for P in Zl:
        for Q in Zl:
            if admissible_edge(P, Q, unit_drop_edges):
                A[(P, Q)] = 1
    return A


# ====== MCMC over Z (potency sets). A is deterministic (all admissible edges). ======
# Initial mcmc map search (no fitch, pure random)
def mcmc_map_search(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str,str]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = False, # Usually False for Z-search (implicit A_full)
    fixed_k: Optional[int] = None,
    steps: int = 5000,
    burn_in: int = 1000,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    block_swap_sizes: Tuple[int, ...] = (1, 2),
    fitch_probs: Dict[FrozenSet[str], float] = None
) -> Tuple["Structure", float, Dict]:
    """
    Fast MCMC sampler using Greedy (Bottom-Up) Scoring.
    
    Logic:
    - Proposes Z sets.
    - Scores them using the 'Greedy' assumption (Child decides best label, Parent must adapt).
    - Much faster than DP, but may reject valid Z sets if the greedy path hits a dead end.
    """
    rng = random.Random(seed)

    # --- Initialization (Unchanged) ---
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
        if not candidate_pool:
            candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    pool_set = set(candidate_pool)

    def make_struct(Zset: Set[FrozenSet[str]]) -> "Structure":
        # In greedy search, Reach isn't strictly used for convolution, 
        # but we need the Structure object for the signature.
        # We assume A_full (fully connected) for Z-search.
        A = _full_edges_for_Z(Zset, unit_drop_edges)
        return Structure(S, Zset, A, unit_drop=unit_drop_edges)

    # Seed Z
    singles = {frozenset([t]) for t in S}
    root = frozenset(S)
    if priors.potency_mode == "fixed_k" and fixed_k is not None:
        try:
            _, Z0 = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
            multis0 = [P for P in Z0 if len(P) >= 2]
            if len(multis0) != fixed_k: raise RuntimeError("Init fail")
        except:
            Z0 = set(singles); Z0.add(root)
            avail = [P for P in candidate_pool if P != root]
            if len(avail) < max(0, fixed_k-1): raise ValueError("Pool too small")
            Z0.update(rng.sample(avail, fixed_k - 1))
    else:
        Z0 = set(singles); Z0.add(root)
        if len(candidate_pool) > 0:
            Z0.update(rng.sample(candidate_pool, min(5, len(candidate_pool))))

    current = make_struct(Z0)
    
    # --- SCORE INITIAL STATE (Greedy) ---
    curr_score = score_structure_greedy_likelihood(current, trees, leaf_type_maps, priors)
    
    if not math.isfinite(curr_score):
        # Fallback: Try 20 random seeds to find a valid starting point
        # Because Greedy is brittle, initialization is harder.
        found = False
        for _ in range(20):
            if priors.potency_mode == "fixed_k":
                Z0 = set(singles); Z0.add(root)
                Z0.update(rng.sample([P for P in candidate_pool if P!=root], fixed_k-1))
            else:
                Z0 = set(singles); Z0.add(root)
                Z0.update(rng.sample(candidate_pool, 3))
            
            current = make_struct(Z0)
            curr_score = score_structure_greedy_likelihood(current, trees, leaf_type_maps, priors)
            if math.isfinite(curr_score):
                found = True
                break
        if not found:
            raise RuntimeError("Could not find a finite-scoring starting point using Greedy Scoring.")

    # --- Helper: Proposals ---
    def propose_fixed_k_swap(Zset):
        acti = [P for P in Zset if len(P) >= 2 and P != root]
        ina = [P for P in candidate_pool if P not in Zset]
        if not acti or not ina: return None, None
        
        # Weighted Swap
        drop = random.choices(acti, weights=[1.0-fitch_probs.get(p, 0.9) for p in acti], k=1)
        add = random.choices(ina, weights=[fitch_probs.get(p, 0.1) for p in ina], k=1)
        
        Z2 = set(Zset)
        Z2.difference_update(drop)
        Z2.update(add)
        return Z2, {'drop': drop, 'add': add}

    def propose_toggle(Zset):
        P = rng.choice(list(pool_set))
        Z2 = set(Zset)
        if P in Z2: Z2.remove(P)
        else: Z2.add(P)
        return Z2

    # --- MCMC Loop ---
    best_struct = current.clone()
    best_score = curr_score
    
    kept_Z = []
    kept_scores = []
    all_scores_trace = [curr_score]
    accepts = 0
    tried = 0
    
    def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (Greedy)", leave=True)

    for it in iterator:
        Zprop = None
        details = None
        
        if priors.potency_mode == "fixed_k":
            Zprop, details = propose_fixed_k_swap(current.Z_active)
        else:
            Zprop = propose_toggle(current.Z_active)

        if Zprop is None:
            tried += 1
            continue

        prop_struct = make_struct(Zprop)
        
        # --- SCORE PROPOSAL (Greedy) ---
        prop_score = score_structure_greedy_likelihood(prop_struct, trees, leaf_type_maps, priors)

        # --- Accept/Reject ---
        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            # Standard Metropolis rule
            if delta >= 0 or rng.random() < math.exp(delta):
                accept = True
        
        tried += 1
        if accept:
            accepts += 1
            current = prop_struct
            curr_score = prop_score
            
            if curr_score > best_score:
                best_score = curr_score
                best_struct = current.clone()
            
            if details and progress:
                 # Optional: Print swap details occasionally
                 pass

        all_scores_trace.append(curr_score)

        # Collect samples
        if it >= burn_in and (it - burn_in) % thin == 0:
            kept_Z.append({P for P in current.Z_active if len(P) >= 2})
            kept_scores.append(curr_score)
            
        if progress:
            iterator.set_postfix({"score": f"{curr_score:.1f}", "best": f"{best_score:.1f}", "acc": f"{accepts/tried:.2f}"})

    # --- Final Stats ---
    counts = {P: 0 for P in pool_set}
    for Zs in kept_Z:
        for P in Zs:
            if P in counts: counts[P] += 1
    
    total_kept = max(1, len(kept_Z))
    inclusion = {P: counts[P] / total_kept for P in counts}
    
    stats = {
        "samples": kept_Z,
        "scores": kept_scores,
        "all_scores_trace": all_scores_trace,
        "accept_rate": accepts / max(1, tried),
        "inclusion": inclusion
    }

    return best_struct, best_score, stats

def mcmc_edges_only_search(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    priors: "Priors",
    Z_fixed: Set[FrozenSet[str]],      # <<< CHANGE: Takes Z_fixed as input now
    *,
    unit_drop_edges: bool = True,      # <<< ADDED: Needed to create the Structure
    steps: int = 2000,
    burn_in: int = 500,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True,
    # position: int = 0
) -> Tuple["Structure", float, Dict]:
    """
    MCMC sampler that explores ONLY the edge space (A) for a FIXED set of
    potencies (Z). It always starts by building a new mid-sized DAG for A.
    """
    rng = random.Random(seed)

    # <<< CHANGE: Always build a fresh mid-sized DAG for the starting A
    initial_A = build_mid_sized_connected_dag(Z_fixed, keep_prob=0.35, rng=rng)
    current = Structure(S, Z_fixed, initial_A, unit_drop=unit_drop_edges)

    # Initial scoring
    curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)
    if not math.isfinite(curr_score):
        raise RuntimeError("Could not find a finite-scoring starting point for edge search.")

    best_struct = current.clone()
    best_score = curr_score

    # ---------- symmetric proposal kernels ----------
    def admissible_pairs(Zset: Set[FrozenSet[str]]) -> List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        L = list(Zset)
        out = []
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, unit_drop_edges):
                    out.append((P, Q))
        return out

    def prop_edge_toggle(st: "Structure", rng: random.Random) -> Tuple[Optional["Structure"], Optional[Dict]]:
        """
        Proposes flipping a single admissible edge on or off.
        Returns the new structure and a dictionary detailing the change.
        """
        pairs = admissible_pairs(st.Z_active)
        if not pairs:
            return None, None
        
        e = rng.choice(pairs)
        new = st.clone()
        
        # Create a dictionary to hold the details of this move
        details = {'edge': e}
        
        # Check if the edge exists in the *current* structure to determine the action
        if st.A.get(e, 0) == 1:
            del new.A[e]
            details['action'] = 'removed'
        else:
            new.A[e] = 1
            details['action'] = 'added'
            
        new.recompute_reach()
        return new, details
    
    # ... the rest of the MCMC loop remains exactly the same ...

    accepts = 0
    tried = 0
    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc=f"Chain (Edges)", leave=True)
    def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

    for it in iterator:
        prop, details = prop_edge_toggle(current, rng)
        if prop is None:
            tried += 1
            continue
        prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors)
        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta))
        tried += 1
        if accept:
            accepts += 1
            if details:
                edge_str = f"{pot_str(details['edge'][0])} -> {pot_str(details['edge'][1])}"
                action_str = details['action'].capitalize()
                # print(f"[Accepted Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")
            current = prop
            curr_score = prop_score
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score
        if progress:
             iterator.set_postfix({
                 "logpost": f"{curr_score:.3f}", "best": f"{best_score:.3f}",
                 "acc": f"{(accepts/max(1,tried)):.2f}", "E": f"{sum(current.A.values())}"
             })

    stats = {"accept_rate": accepts / max(1, tried)}
    return best_struct, best_score, stats

# ====== MCMC with edges kept in the target (log P(Z) + log P(A|Z) + Σ log P(T|Z,A)) ======

def mcmc_map_search_with_edges(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,          # pass k when priors.potency_mode == "fixed_k"
    steps: int = 6000,
    burn_in: int = 1500,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True,
    # move mixture: probability of choosing a potency move vs an edge move
    p_potency_move: float = 0.5,
    # for fixed-k swaps: sizes of swap blocks to try
    block_swap_sizes: Tuple[int, ...] = (1, 2),
    # optional compact pool for multi-type potencies (recommended: collect_fitch_multis)
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
) -> Tuple["Structure", float, Dict]:
    """
    Samples over BOTH Z and A.
      - Potency moves:
          * fixed_k: swap m active multis with m inactive (m in block_swap_sizes)
          * bernoulli: toggle a candidate multi on/off
        (Edges incident to dropped potencies are pruned; no auto-wiring for added potencies.)
      - Edge moves:
          * toggle a single admissible ordered pair (P->Q) on/off (uniform over admissible pairs)

    All proposals are symmetric ==> MH accept prob = min(1, exp(delta_log_posterior)).

    Returns:
      best_struct, best_score, stats (accept_rate, inclusion freqs, edge density trace, etc.)
    """
    rng = random.Random(seed)

    # ---------- candidate pool for multis ----------
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
        if not candidate_pool:
            candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    pool_set = set(candidate_pool)

    singles = {frozenset([t]) for t in S}
    root = frozenset(S)

    # ---------- seed state ----------
    def seed_Z() -> Set[FrozenSet[str]]:
        if priors.potency_mode == "fixed_k" and fixed_k is not None:
            try:
                _, Zf = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
                multis0 = [P for P in Zf if len(P) >= 2]
                if len(multis0) == fixed_k:
                    return set(Zf)
            except Exception:
                pass
            # fallback uniform from pool
            Z0 = set(singles); Z0.add(root)
            avail = [P for P in candidate_pool if P != root]
            if len(avail) < max(0, fixed_k - 1):
                raise RuntimeError("Not enough candidates to seed fixed-k MCMC.")
            Z0.update(rng.sample(avail, fixed_k - 1))
            return Z0
        else:
            Z0 = set(singles); Z0.add(root)
            sprinkle = min(5, len(candidate_pool))
            if sprinkle:
                Z0.update(rng.sample(candidate_pool, sprinkle))
            return Z0

    def seed_A(Z: Set[FrozenSet[str]]) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
        # any reasonable initial A works; we reuse your helper
        return build_mid_sized_connected_dag(Z, keep_prob=0.35, rng=rng)

    Z = seed_Z()
    A = seed_A(Z)
    current = Structure(S, Z, A, unit_drop=unit_drop_edges)
    curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)
    if not math.isfinite(curr_score):
        # rescue a few times
        for _ in range(20):
            Z = seed_Z(); A = seed_A(Z)
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)
            if math.isfinite(curr_score):
                break
        if not math.isfinite(curr_score):
            raise RuntimeError("Could not find a finite-scoring starting point for MCMC.")

    best_struct = current.clone()
    best_score = curr_score

    # ---------- symmetric proposal kernels ----------
    def admissible_pairs(Zset: Set[FrozenSet[str]]) -> List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        L = list(Zset)
        out = []
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, unit_drop_edges):
                    out.append((P, Q))
        return out
    
    def prop_edge_toggle(st: "Structure", rng: random.Random) -> Tuple[Optional["Structure"], Optional[Dict]]:
        """
        Proposes flipping a single admissible edge on or off.
        Returns the new structure and a dictionary detailing the change.
        """
        pairs = admissible_pairs(st.Z_active)
        if not pairs:
            return None, None
        
        e = rng.choice(pairs)
        new = st.clone()
        
        # Create a dictionary to hold the details of this move
        details = {'edge': e}
        
        # Check if the edge exists in the *current* structure to determine the action
        if st.A.get(e, 0) == 1:
            del new.A[e]
            details['action'] = 'removed'
        else:
            new.A[e] = 1
            details['action'] = 'added'
            
        new.recompute_reach()
        return new, details
    

    def prop_potency_fixed_k(st: "Structure",
                         rng: random.Random,
                         candidate_pool: List[FrozenSet[str]],
                         block_swap_sizes: Tuple[int, ...]) -> Tuple[Optional["Structure"], Optional[Dict]]:
        """
        Proposes a new structure by swapping 'm' potencies.
        It prunes old edges, then intelligently rewires the new potencies by:
        1. Adding a random subset (35%) of all possible new edges.
        2. Forcibly adding edges if needed to ensure the new potency is connected
            to the main graph (a path exists from the root, through the new
            potency, to a leaf).

        Returns:
            A tuple of (new_structure, details_of_the_swap).
        """
        act = [P for P in st.Z_active if len(P) >= 2]
        ina = [P for P in candidate_pool if P not in st.Z_active]
        if not act or not ina:
            return None, None

        m = rng.choice(block_swap_sizes)
        m = max(1, min(m, len(act), len(ina)))
        drop = rng.sample(act, m)
        add = rng.sample(ina, m)

        new = st.clone()

        # 1. Remove potencies and prune their incident edges
        for P in drop:
            new.Z_active.remove(P)
        if new.A:
            new.A = {e: v for e, v in new.A.items() if (e[0] not in drop and e[1] not in drop)}

        # 2. Add the new potencies
        for P in add:
            new.Z_active.add(P)

        # 3. Rewire the newly added potencies
        for P_new in add:
            # Find all potential parents and children for the new potency
            admissible_parents = [p for p in new.Z_active if admissible_edge(p, P_new, new.unit_drop)]
            admissible_children = [c for c in new.Z_active if admissible_edge(P_new, c, new.unit_drop)]

            # 3a. Add a random subset of possible edges
            for parent in admissible_parents:
                if rng.random() < 0.35:
                    new.A[(parent, P_new)] = 1
            for child in admissible_children:
                if rng.random() < 0.35:
                    new.A[(P_new, child)] = 1

            # 3b. Enforce connectivity if the random wiring missed it
            has_incoming_edge = any(e[1] == P_new for e in new.A)
            if not has_incoming_edge and admissible_parents:
                # Forcibly connect it from a random parent to ensure a path from the root
                chosen_parent = rng.choice(admissible_parents)
                new.A[(chosen_parent, P_new)] = 1

            has_outgoing_edge = any(e[0] == P_new for e in new.A)
            if not has_outgoing_edge and admissible_children:
                # Forcibly connect it to a random child to ensure a path to a leaf
                chosen_child = rng.choice(admissible_children)
                new.A[(P_new, chosen_child)] = 1

        new.recompute_reach()
    
        # Return both the new structure and the details of the swap for logging
        swap_details = {'drop': drop, 'add': add}
        return new, swap_details

    def prop_potency_toggle(st: "Structure") -> Optional["Structure"]:
        # toggle a random candidate multi (root/singletons are never toggled)
        P = rng.choice(candidate_pool)
        new = st.clone()
        if P in new.Z_active:
            new.Z_active.remove(P)
            # prune incident edges
            if new.A:
                new.A = {e: v for e, v in new.A.items() if (P not in e)}
        else:
            new.Z_active.add(P)
            # no auto-wiring (edges will be explored separately)
        new.recompute_reach()
        return new

    # ---------- MCMC main loop ----------
    kept_Z = []
    kept_scores = []
    kept_edge_density = []

    accepts = 0
    tried = 0

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (Z & A)", leave=True)

    for it in iterator:
        prop = None
        last_proposal_details = None
        do_potency = (rng.random() < p_potency_move)

        if do_potency:
            if priors.potency_mode == "fixed_k" and fixed_k is not None:
                # prop = prop_potency_fixed_k(current)
                prop, last_proposal_details = prop_potency_fixed_k(current, rng, candidate_pool, block_swap_sizes)
            else:
                prop = prop_potency_toggle(current)
        else:
            prop, last_proposal_details = prop_edge_toggle(current, rng)
            # prop = prop_edge_toggle(current)

        if prop is None:
            # count as tried but no change
            tried += 1
            if progress:
                iterator.set_postfix({
                    "logpost": f"{curr_score:.3f}",
                    "best": f"{best_score:.3f}",
                    "acc": f"{(accepts/max(1,tried)):.2f}",
                    "E": f"{sum(prop.A.values()) if prop else sum(current.A.values())}"
                })
            continue

        prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors)

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta))

        tried += 1
        if accept:
            accepts += 1

            if last_proposal_details and 'drop' in last_proposal_details:
                # This was a successful swap move, so we print the details
                dropped_str = ", ".join(pot_str(p) for p in last_proposal_details['drop'])
                added_str = ", ".join(pot_str(p) for p in last_proposal_details['add'])
                # print(f"[Accepted Swap]: {dropped_str} -> {added_str} | New Score: {prop_score:.3f}")

            elif last_proposal_details and 'edge' in last_proposal_details:
                edge_details = last_proposal_details
                edge_str = f"{pot_str(edge_details['edge'][0])} -> {pot_str(edge_details['edge'][1])}"
                action_str = edge_details['action'].capitalize()
                # print(f"[Accepted Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")

            current = prop
            curr_score = prop_score
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score

        # collect sample
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_Z.append({P for P in current.Z_active if len(P) >= 2})
            kept_scores.append(curr_score)
            kept_edge_density.append(sum(v for v in current.A.values()))

        if progress:
            iterator.set_postfix({
                "logpost": f"{curr_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc": f"{(accepts/max(1,tried)):.2f}",
                "E": f"{sum(current.A.values())}"
            })

    # posterior inclusion frequency for potencies in pool
    counts = {P: 0 for P in pool_set}
    for Zs in kept_Z:
        for P in Zs:
            if P in counts:
                counts[P] += 1
    total_kept = max(1, len(kept_Z))
    inclusion = {P: counts[P] / total_kept for P in counts}

    stats = {
        "samples_Z": kept_Z,
        "scores": kept_scores,
        "edge_density": kept_edge_density,
        "accept_rate": accepts / max(1, tried),
        "inclusion": inclusion,
    }
    return best_struct, best_score, stats

def mcmc_map_search_only_A(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,          # pass k when priors.potency_mode == "fixed_k"
    steps: int = 6000,
    burn_in: int = 1500,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True,
    # move mixture: probability of choosing a potency move vs an edge move
    # p_potency_move: float = 0.5,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,      # optional compact pool for multi-type potencies (recommended: collect_fitch_multis)
    # for fixed-k swaps: sizes of swap blocks to try
    block_swap_sizes: Tuple[int, ...] = (1, 2),
    Z
) -> Tuple["Structure", float, Dict]:
    """
    Samples over BOTH Z and A.
      - Potency moves:
          * fixed_k: swap m active multis with m inactive (m in block_swap_sizes)
          * bernoulli: toggle a candidate multi on/off
        (Edges incident to dropped potencies are pruned; no auto-wiring for added potencies.)
      - Edge moves:
          * toggle a single admissible ordered pair (P->Q) on/off (uniform over admissible pairs)

    All proposals are symmetric ==> MH accept prob = min(1, exp(delta_log_posterior)).

    Returns:
      best_struct, best_score, stats (accept_rate, inclusion freqs, edge density trace, etc.)
    """
    rng = random.Random(seed)

    # ---------- candidate pool for multis ----------
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
        if not candidate_pool:
            candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    pool_set = set(candidate_pool)

    singles = {frozenset([t]) for t in S}
    root = frozenset(S)

    def seed_A(Z: Set[FrozenSet[str]]) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
        # any reasonable initial A works; we reuse your helper
        return build_mid_sized_connected_dag(Z, keep_prob=0.35, rng=rng)

    # Z = seed_Z()
    A = seed_A(Z)
    current = Structure(S, Z, A, unit_drop=unit_drop_edges)
    curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)
    if not math.isfinite(curr_score):
        # rescue a few times
        for _ in range(20):
            # Z = seed_Z(); 
            A = seed_A(Z)
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)
            if math.isfinite(curr_score):
                break
        if not math.isfinite(curr_score):
            raise RuntimeError("Could not find a finite-scoring starting point for MCMC.")

    best_struct = current.clone()
    best_score = curr_score

    # ---------- symmetric proposal kernels ----------
    def admissible_pairs(Zset: Set[FrozenSet[str]]) -> List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        L = list(Zset)
        out = []
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, unit_drop_edges):
                    out.append((P, Q))
        return out
    
    def prop_edge_toggle(st: "Structure", rng: random.Random) -> Tuple[Optional["Structure"], Optional[Dict]]:
        """
        Proposes flipping a single admissible edge on or off.
        Returns the new structure and a dictionary detailing the change.
        """
        pairs = admissible_pairs(st.Z_active)
        if not pairs:
            return None, None
        
        e = rng.choice(pairs)
        new = st.clone()
        
        # Create a dictionary to hold the details of this move
        details = {'edge': e}
        
        # Check if the edge exists in the *current* structure to determine the action
        if st.A.get(e, 0) == 1:
            del new.A[e]
            details['action'] = 'removed'
        else:
            new.A[e] = 1
            details['action'] = 'added'
            
        new.recompute_reach()
        return new, details

    def prop_potency_fixed_k(st: "Structure",
                         rng: random.Random,
                         candidate_pool: List[FrozenSet[str]],
                         block_swap_sizes: Tuple[int, ...]) -> Tuple[Optional["Structure"], Optional[Dict]]:
        """
        Proposes a new structure by swapping 'm' potencies.
        It prunes old edges, then intelligently rewires the new potencies by:
        1. Adding a random subset (35%) of all possible new edges.
        2. Forcibly adding edges if needed to ensure the new potency is connected
            to the main graph (a path exists from the root, through the new
            potency, to a leaf).

        Returns:
            A tuple of (new_structure, details_of_the_swap).
        """
        act = [P for P in st.Z_active if len(P) >= 2]
        ina = [P for P in candidate_pool if P not in st.Z_active]
        if not act or not ina:
            return None, None

        m = rng.choice(block_swap_sizes)
        m = max(1, min(m, len(act), len(ina)))
        drop = rng.sample(act, m)
        add = rng.sample(ina, m)

        new = st.clone()

        # 1. Remove potencies and prune their incident edges
        for P in drop:
            new.Z_active.remove(P)
        if new.A:
            new.A = {e: v for e, v in new.A.items() if (e[0] not in drop and e[1] not in drop)}

        # 2. Add the new potencies
        for P in add:
            new.Z_active.add(P)

        # 3. Rewire the newly added potencies
        for P_new in add:
            # Find all potential parents and children for the new potency
            admissible_parents = [p for p in new.Z_active if admissible_edge(p, P_new, new.unit_drop)]
            admissible_children = [c for c in new.Z_active if admissible_edge(P_new, c, new.unit_drop)]

            # 3a. Add a random subset of possible edges
            for parent in admissible_parents:
                if rng.random() < 0.35:
                    new.A[(parent, P_new)] = 1
            for child in admissible_children:
                if rng.random() < 0.35:
                    new.A[(P_new, child)] = 1

            # 3b. Enforce connectivity if the random wiring missed it
            has_incoming_edge = any(e[1] == P_new for e in new.A)
            if not has_incoming_edge and admissible_parents:
                # Forcibly connect it from a random parent to ensure a path from the root
                chosen_parent = rng.choice(admissible_parents)
                new.A[(chosen_parent, P_new)] = 1

            has_outgoing_edge = any(e[0] == P_new for e in new.A)
            if not has_outgoing_edge and admissible_children:
                # Forcibly connect it to a random child to ensure a path to a leaf
                chosen_child = rng.choice(admissible_children)
                new.A[(P_new, chosen_child)] = 1

        new.recompute_reach()
    
        # Return both the new structure and the details of the swap for logging
        swap_details = {'drop': drop, 'add': add}
        return new, swap_details

    def prop_potency_toggle(st: "Structure") -> Optional["Structure"]:
        # toggle a random candidate multi (root/singletons are never toggled)
        P = rng.choice(candidate_pool)
        new = st.clone()
        if P in new.Z_active:
            new.Z_active.remove(P)
            # prune incident edges
            if new.A:
                new.A = {e: v for e, v in new.A.items() if (P not in e)}
        else:
            new.Z_active.add(P)
            # no auto-wiring (edges will be explored separately)
        new.recompute_reach()
        return new

    # ---------- MCMC main loop ----------
    kept_Z = []
    kept_scores = []
    kept_edge_density = []

    accepts = 0
    tried = 0

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (A-only)", leave=True)

    for it in iterator:
        prop = None
        last_proposal_details = None
        
        prop, last_proposal_details = prop_edge_toggle(current, rng)
        # prop = prop_edge_toggle(current)

        if prop is None:
            # count as tried but no change
            tried += 1
            if progress:
                iterator.set_postfix({
                    "logpost": f"{curr_score:.3f}",
                    "best": f"{best_score:.3f}",
                    "acc": f"{(accepts/max(1,tried)):.2f}",
                    "E": f"{sum(prop.A.values()) if prop else sum(current.A.values())}"
                })
            continue

        prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors)

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta))

        tried += 1
        if accept:
            accepts += 1

            if last_proposal_details and 'edge' in last_proposal_details:
                edge_details = last_proposal_details
                edge_str = f"{pot_str(edge_details['edge'][0])} -> {pot_str(edge_details['edge'][1])}"
                action_str = edge_details['action'].capitalize()
                # print(f"[Accepted Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")

            current = prop
            curr_score = prop_score
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score

        # collect sample
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_Z.append({P for P in current.Z_active if len(P) >= 2})
            kept_scores.append(curr_score)
            kept_edge_density.append(sum(v for v in current.A.values()))

        if progress:
            iterator.set_postfix({
                "logpost": f"{curr_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc": f"{(accepts/max(1,tried)):.2f}",
                "E": f"{sum(current.A.values())}"
            })

    # posterior inclusion frequency for potencies in pool
    counts = {P: 0 for P in pool_set}
    for Zs in kept_Z:
        for P in Zs:
            if P in counts:
                counts[P] += 1
    total_kept = max(1, len(kept_Z))
    inclusion = {P: counts[P] / total_kept for P in counts}

    stats = {
        "samples_Z": kept_Z,
        "scores": kept_scores,
        "edge_density": kept_edge_density,
        "accept_rate": accepts / max(1, tried),
        "inclusion": inclusion,
    }
    return best_struct, best_score, stats


def _mcmc_worker_A(args: tuple):
    """
    A simple worker function that unpacks arguments and calls the MCMC sampler.
    This is the target for each parallel process.
    """
    # Unpack all the arguments passed by the main parallel function
    (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, candidate_pool, block_swap_sizes, Z) = args

    # Call the MCMC sampler with the unique seed for this worker
    # Note: We disable the progress bar for worker processes to keep the console clean
    return mcmc_map_search_only_A(
        S=S, trees=trees, leaf_type_maps=leaf_type_maps, priors=priors,
        unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
        burn_in=burn_in, thin=thin, seed=seed, progress=True,
        candidate_pool=candidate_pool, block_swap_sizes=block_swap_sizes, Z = Z
    )

def _mcmc_worker_Z(args: tuple):
    """
    A simple worker function that unpacks arguments and calls the MCMC sampler.
    This is the target for each parallel process.
    """
    # Unpack all the arguments passed by the main parallel function
    (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, candidate_pool, block_swap_sizes, fitch_probs) = args

    # Call the MCMC sampler with the unique seed for this worker
    # Note: We disable the progress bar for worker processes to keep the console clean
    return mcmc_map_search(
        S=S, trees=trees, leaf_type_maps=leaf_type_maps, priors=priors,
        unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
        burn_in=burn_in, thin=thin, seed=seed, progress=True,
        candidate_pool=candidate_pool, block_swap_sizes=block_swap_sizes, fitch_probs = fitch_probs
    )


def run_mcmc_only_Z_parallel(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    steps: int = 5000,
    burn_in: int = 1000,
    thin: int = 10,
    base_seed: int = 123,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    block_swap_sizes: Tuple[int, ...] = (1, 2),
    n_chains: int = 4,
    num_cores: int = 7,
    fitch_probs: Optional[Dict[FrozenSet[str], float]] = None # <<<--- ADD THIS ARGUMENT
):
    """
    Runs multiple MCMC chains in parallel and returns the best result found.
    """
    if n_chains <= 0:
        n_chains = max(1, os.cpu_count() - 1)

    # Generate unique seeds for each independent chain
    seeds = [base_seed + i for i in range(n_chains)]

    # Package arguments for each worker. Each worker gets a different seed.
    tasks = []
    for seed in seeds:
        tasks.append((
            S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
            steps, burn_in, thin, seed, True, candidate_pool, block_swap_sizes, fitch_probs
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel MCMC chains...")
    with ProcessPoolExecutor(max_workers=min(n_chains, min(num_cores, os.cpu_count() - 1))) as executor:
        futures = [executor.submit(_mcmc_worker_Z, t) for t in tasks]

        for future in as_completed(futures):
            try:
                # Unpack results from a completed chain
                struct, score, stats = future.result()
                all_stats.append(stats)

                # Keep track of the single best solution found across all chains
                if score > best_score:
                    best_score = score
                    best_struct = struct

            except Exception as e:
                print(f"A chain failed with an error: {e}")

    print(f"[Info] All chains finished. Best score found: {best_score:.4f}")
    return best_struct, best_score, all_stats


# ========= helpers: joint swap + rewire =========

def _admissible_supersets(P: FrozenSet[str], Z_active: Set[FrozenSet[str]], unit_drop: bool):
    # parents R -> P
    return [R for R in Z_active if admissible_edge(R, P, unit_drop)]

def _admissible_subsets(P: FrozenSet[str], Z_active: Set[FrozenSet[str]], unit_drop: bool):
    # children P -> Q
    return [Q for Q in Z_active if admissible_edge(P, Q, unit_drop)]

def propose_swap_and_rewire(current: "Structure",
                            rng: random.Random,
                            m: int = 2,
                            keep_prob: float = 0.5) -> Optional["Structure"]:
    """
    Fixed-k friendly: swap out m multis not in Z and add m new ones.
    In the *same* proposal, rewire edges locally around the added potencies.
    """
    multis = [P for P in current.Z_active if len(P) >= 2]
    pool_add = [P for P in current.potencies_multi_all() if P not in current.Z_active]
    if not multis or not pool_add:
        return None

    m = max(1, min(m, len(multis), len(pool_add)))
    to_remove = rng.sample(multis, m)
    to_add    = rng.sample(pool_add, m)

    new = current.clone()

    # Remove selected potencies and their incident edges
    for P in to_remove:
        if P in new.Z_active:
            new.Z_active.remove(P)
        if new.A:
            new.A = {e: v for e, v in new.A.items() if P not in e}

    # Add potencies and wire locally
    for P in to_add:
        new.Z_active.add(P)

        supers  = _admissible_supersets(P, new.Z_active, new.unit_drop)
        subsets = _admissible_subsets(P, new.Z_active, new.unit_drop)

        for R in supers:
            if rng.random() < keep_prob:
                new.A[(R, P)] = 1
        for Q in subsets:
            if rng.random() < keep_prob:
                new.A[(P, Q)] = 1

        # ensure at least one parent/child if available
        if not any(e for e in new.A if e[1] == P) and supers:
            new.A[(rng.choice(supers), P)] = 1
        if not any(e for e in new.A if e[0] == P) and subsets:
            new.A[(P, rng.choice(subsets))] = 1

    new.recompute_reach()
    return new

# ========= helpers: candidate multis from Fitch labels (small pool) =========

def collect_fitch_multis(S: List[str],
                         trees: List["TreeNode"],
                         leaf_type_maps: List[Dict[str, str]]) -> List[FrozenSet[str]]:
    """
    Build a *compact* candidate pool of multi-type potencies by running union-Fitch
    on each tree and collecting the node potencies (len>=2).
    """
    pool = set()
    for tree, m in zip(trees, leaf_type_maps):
        assign_union_potency(tree, m)
        # walk nodes
        stack=[tree]
        while stack:
            v=stack.pop()
            for c in v.children: stack.append(c)
            if hasattr(v, "potency"):
                P = frozenset(v.potency)
                if len(P) >= 2:
                    pool.add(P)
    # also include full root
    pool.add(frozenset(S))
    return sorted(list(pool), key=lambda x: (len(x), tuple(sorted(list(x)))))

# ========= helpers: GRASP seed (greedy randomized constructive) =========
def grasp_seed(
    S,
    trees,
    leaf_type_maps,
    priors,
    k,  # number of multi-type potencies (fixed_k)
    candidates: Optional[List[FrozenSet[str]]] = None,
    alpha: float = 0.3,         # 0=greedy, 1=random
    iters_local: int = 20,
    unit_drop_edges: bool = True,
    rng: Optional[random.Random] = None,
    sample_per_step: int = 200  # evaluate at most this many candidates per add
):
    """
    GRASP constructor for a seed structure:
      • Builds Z starting from ROOT + singletons and adds (k-1) multi-type potencies.
      • Uses a RELAXED (Bernoulli) prior while Z is partial (to avoid -inf under fixed-k).
      • After Z is built, wires a mid-density DAG and does a small local edge improvement.

    Returns:
        (Structure, score)
    """
    if rng is None:
        rng = random.Random(0)

    # Start with ROOT (all types) + all singletons
    root = frozenset(S)
    Z = {frozenset([t]) for t in S}
    Z.add(root)

    # Candidate pool for multis
    if candidates is None:
        candidates = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    else:
        candidates = [P for P in candidates if len(P) >= 2]

    # Relaxed prior for partial Z (avoid -inf before we reach exactly k multis)
    relaxed_priors = Priors(potency_mode="bernoulli", pi_P=0.5, rho=priors.rho)

    def score_Z(Zset: Set[FrozenSet[str]]) -> float:
        """Score a candidate Z with relaxed prior until it has k multis; then real prior."""
        multis_now = sum(1 for P in Zset if len(P) >= 2)
        pri = priors if multis_now >= k else relaxed_priors
        A0 = build_mid_sized_connected_dag(Zset, keep_prob=0.35, rng=rng)
        st  = Structure(S, Zset, A0, unit_drop=unit_drop_edges)
        sc, _ = score_structure(st, trees, leaf_type_maps, pri)
        return sc

    _ = score_Z(Z)  # prime cache / sanity

    # Add k-1 multi-type potencies
    for _ in range(max(0, k - 1)):
        pool = [P for P in candidates if P not in Z]
        if not pool:
            break

        rng.shuffle(pool)
        pool = pool[:min(sample_per_step, len(pool))]

        # Score sampled candidates
        scored: List[Tuple[float, FrozenSet[str]]] = []
        for P in pool:
            Ztry = set(Z); Ztry.add(P)
            sc = score_Z(Ztry)
            scored.append((sc, P))

        # Keep only finite scores; fallback to random if empty
        finite = [(sc, P) for sc, P in scored if math.isfinite(sc)]
        if not finite:
            P_pick = rng.choice(pool)  # still move forward
        else:
            finite.sort(reverse=True, key=lambda x: x[0])
            best_sc, worst_sc = finite[0][0], finite[-1][0]
            if (not math.isfinite(best_sc)) or (not math.isfinite(worst_sc)) or (best_sc == worst_sc):
                rcl = [P for _, P in finite]
            else:
                cut = best_sc - alpha * (best_sc - worst_sc)
                rcl = [P for sc, P in finite if sc >= cut]
            if not rcl:
                rcl = [P for _, P in finite]
            P_pick = rng.choice(rcl)

        Z.add(P_pick)

    # Build edges + quick local edge improvement under the REAL prior
    A = build_mid_sized_connected_dag(Z, keep_prob=0.4, rng=rng)
    current = Structure(S, Z, A, unit_drop=unit_drop_edges)
    curr_score, _ = score_structure(current, trees, leaf_type_maps, priors)

    for _ in range(iters_local):
        improved = False

        prop = current.propose_add_edge(rng)
        if prop is not None:
            s, _ = score_structure(prop, trees, leaf_type_maps, priors)
            if s > curr_score:
                current, curr_score = prop, s
                improved = True

        prop = current.propose_remove_edge(rng)
        if prop is not None:
            s, _ = score_structure(prop, trees, leaf_type_maps, priors)
            if s > curr_score:
                current, curr_score = prop, s
                improved = True

        if not improved:
            break

    return current, curr_score


# ========= helpers: Large Neighborhood Search (destroy & repair) =========

def lns_destroy_repair(current: "Structure",
                       trees, leaf_type_maps, priors,
                       destroy_frac: float = 0.3,
                       repair_k: Optional[int] = None,
                       rng: Optional[random.Random] = None,
                       sample_per_step: int = 200):
    if rng is None:
        rng = random.Random()
    multis = [P for P in current.Z_active if len(P) >= 2]
    if not multis:
        return None

    m = max(1, int(round(destroy_frac * len(multis))))
    to_drop = rng.sample(multis, m)

    # drop them
    Z = set(current.Z_active)
    for P in to_drop:
        Z.remove(P)
    A = {e: v for e, v in current.A.items() if (e[0] not in to_drop and e[1] not in to_drop)}

    target_k = repair_k if repair_k is not None else len(multis)
    need = max(0, target_k - len([P for P in Z if len(P) >= 2]))

    # candidate pool from Fitch-derived labels (keeps it small)
    # NOTE: caller should pass consistent pool for speed, but we can re-collect here if needed
    # pool = [P for P in current.potencies_multi_all() if P not in Z and len(P) >= 2]  # can be huge
    # To stay safe, derive from existing labels neighborhood:
    pool = []
    for P in list(Z):
        if len(P) >= 2:
            # neighbors by 1-element add/remove (if unit_drop edges are used)
            # but ensure we only consider multis
            for t in current.S:
                # add one element to form a superset
                if t not in P:
                    Q = frozenset(set(P) | {t})
                    if len(Q) >= 2 and Q not in Z:
                        pool.append(Q)
                # remove one element to form a subset (still multi)
                # (subsets will be length-1 a lot; filter)
    # Dedup + cap
    pool = list({p for p in pool if len(p) >= 2})
    rng.shuffle(pool)

    def score_Z(Zset):
        A0 = build_mid_sized_connected_dag(Zset, keep_prob=0.4, rng=rng)
        st = Structure(current.S, Zset, A0, unit_drop=current.unit_drop)
        sc, _ = score_structure(st, trees, leaf_type_maps, priors)
        return sc, st

    bestZ = set(Z)
    best_sc, best_st = score_Z(bestZ)

    for _ in range(need):
        cand = [P for P in pool if P not in bestZ]
        if not cand:
            break
        rng.shuffle(cand)
        cand = cand[:min(sample_per_step, len(cand))]
        best_inc = None
        for P in cand:
            Ztry = set(bestZ); Ztry.add(P)
            sc_try, st_try = score_Z(Ztry)
            if best_inc is None or sc_try > best_inc[0]:
                best_inc = (sc_try, P, st_try)
        if best_inc is None:
            break
        best_sc, chosenP, best_st = best_inc
        bestZ.add(chosenP)

    return best_st

def read_leaf_type_map_tls(path: str) -> Dict[str, str]:
    """
    Read a leaf->type mapping from a file. Handles JSON and TSV/CSV/TXT.

    For TSV/CSV/TXT (like the multi-column example):
    - Assumes the second column is the leaf identifier (cell barcode).
    - Assumes the third column is the cell type/state.
    - Automatically detects the delimiter (tab or comma).
    - Attempts to automatically detect and skip a header row.
    - Ignores columns beyond the third one.
    - **Filters out leaves with types "Unknown", "aPSM", and "pPSM".**

    Returns: dict {leaf_name: type_symbol} (types are coerced to str)
    """

    ext = os.path.splitext(path)[1].lower()

    # --- JSON Handling ---
    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: JSON must be an object mapping leaf->type.")
        # Store intermediate result before filtering
        out = {str(k): str(v) for k, v in data.items()}

    # --- TSV/CSV/TXT Handling ---
    elif ext in (".csv", ".tsv", ".txt"):
        out = {}
        with open(path, "r", newline="") as f:
            try:
                sample = f.read(2048)
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
                f.seek(0)
                reader = csv.reader(f, dialect)
                # print(f"Detected delimiter '{dialect.delimiter}' for {path}") # Optional: uncomment for debugging
            except csv.Error:
                # print(f"Warning: Could not sniff delimiter for {path}, defaulting to tab.") # Optional: uncomment for debugging
                f.seek(0)
                reader = csv.reader(f, delimiter='\t')

            rows = list(reader)
            if not rows:
                raise ValueError(f"{path}: empty file")

            start_idx = 0
            first_row_cells = [cell.strip().lower() for cell in rows[0]]
            looks_like_header = False
            if len(first_row_cells) >= 3:
                 if (first_row_cells[1] in ['cellbc', 'leaf', 'id', 'barcode']) and \
                    (first_row_cells[2] in ['cell_state', 'type', 'state', 'celltype']):
                     looks_like_header = True

            if looks_like_header:
                # print(f"Detected header in {path}: {rows[0]}") # Optional: uncomment for debugging
                start_idx = 1
            # else:
                 # print(f"No header detected (or assuming no header) in {path}") # Optional: uncomment for debugging

            for i in range(start_idx, len(rows)):
                row = rows[i]
                if len(row) < 3:
                    # print(f"Warning: Skipping row {i+1} in {path}: needs at least 3 columns (index, leaf, type). Content: {row}") # Optional: uncomment for debugging
                    continue

                leaf = row[1].strip().strip('"\'')
                typ = row[2].strip().strip('"\'')

                if not leaf or not typ:
                    # print(f"Warning: Skipping row {i+1} in {path}: has empty leaf (col 2) or type (col 3). Content: {row}") # Optional: uncomment for debugging
                    continue

                if leaf in out:
                    # print(f"Warning: Duplicate leaf '{leaf}' found at line {i+1} in {path}. Overwriting type.") # Optional: uncomment for debugging
                    pass # Allow overwrite silently or warn
                out[leaf] = str(typ)

        if not out:
             # This check is now before filtering, might still return empty if all leaves are excluded later
             print(f"Warning: No valid leaf-type pairs extracted initially from {path}, check format/columns.")
             # raise ValueError(f"No valid leaf-type pairs extracted from {path}")

    else:
        raise ValueError(f"Unsupported mapping file type: {path} (use .csv, .tsv, .txt, or .json)")

    # --- ** FILTERING STEP ** ---
    excluded_types = {"Unknown", "aPSM", "pPSM"}
    filtered_out = {leaf: typ for leaf, typ in out.items() if typ not in excluded_types}

    if not filtered_out and out: # If we started with data but filtered everything
        print(f"Warning: All leaves in {path} were filtered out (types were Unknown, aPSM, or pPSM).")
    # elif len(out) != len(filtered_out):
        # print(f"Filtered {len(out) - len(filtered_out)} leaves (types: Unknown, aPSM, pPSM) from {path}.")


    return filtered_out # Return the filtered dictionary

def read_leaf_type_map(path: str) -> Dict[str, str]:
    """
    Read a leaf->type mapping from a file.

    Supported:
      - JSON dict: { "LeafName": "Type", ... }
      - CSV/TSV/TXT with 2 columns (header optional):
          * If header present, typical field names could be:
              - leaf, type
              - cellBC, cell_state (your .txt example)
    Returns: dict {leaf_name: type_symbol} (types are coerced to str)
    """

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: JSON must be an object mapping leaf->type.")
        return {str(k): str(v) for k, v in data.items()}

    elif ext in (".csv", ".tsv", ".txt"):
        # treat .txt as TSV by default (your example is tab-delimited)
        delim = "\t" if ext in (".tsv", ".txt") else ","
        out = {}
        with open(path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            rows = list(reader)
            if not rows:
                raise ValueError(f"{path}: empty file")

            # Detect header
            start_idx = 0
            header = [h.strip().lower() for h in rows[0]] if rows and rows[0] else []
            has_header = False
            if len(header) >= 2:
                # Common header names we accept
                if ("leaf" in header[0] or "cellbc" in header[0]) and ("type" in header[1] or "cell_state" in header[1]):
                    has_header = True
                # Or any header line where at least one of ('leaf','cellbc') and one of ('type','cell_state') appear
                if not has_header:
                    left_has = any(x in header for x in ("leaf", "cellbc"))
                    right_has = any(x in header for x in ("type", "cell_state"))
                    has_header = left_has and right_has

            if has_header:
                start_idx = 1

            for i in range(start_idx, len(rows)):
                row = rows[i]
                if len(row) < 2:
                    raise ValueError(f"{path}: line {i+1} needs at least 2 columns (leaf,type)")
                leaf = row[0].strip()
                typ  = row[1].strip()
                if not leaf or not typ:
                    raise ValueError(f"{path}: line {i+1} has empty leaf/type")
                if leaf in out:
                    raise ValueError(f"{path}: duplicate leaf '{leaf}' at line {i+1}")
                out[leaf] = str(typ)  # coerce types to string (handles negatives like -7, -9)
        return out

    else:
        raise ValueError(f"Unsupported mapping file type: {path} (use .csv, .tsv, .txt, or .json)")
        

def validate_leaf_type_map(root: TreeNode, leaf_map: Dict[str,str], S: List[str]) -> None:
    """
    Ensure mapping covers exactly the leaves in the tree, and types are in S.
    Raises ValueError if not valid.
    """
    leaves_in_tree = set(collect_leaf_names(root))
    leaves_in_map  = set(leaf_map.keys())

    missing = leaves_in_tree - leaves_in_map
    extra   = leaves_in_map  - leaves_in_tree
    if missing:
        raise ValueError(f"Leaf map missing leaves: {sorted(missing)}")
    if extra:
        raise ValueError(f"Leaf map has unknown leaves not in tree: {sorted(extra)}")

    allowed = set(S)
    bad_types = {t for t in leaf_map.values() if t not in allowed}
    if bad_types:
        raise ValueError(f"Leaf map contains types not in S={S}: {sorted(bad_types)}")


def filter_leaf_map_to_tree(root: TreeNode, leaf_map: Dict[str, str]) -> Dict[str, str]:
    leaves = set(collect_leaf_names(root))
    return {leaf: str(typ) for leaf, typ in leaf_map.items() if leaf in leaves}
    

def _read_json_objects_exact(path: str):
    """Read one JSON object per non-empty line (your file format)."""
    objs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    if not objs:
        raise ValueError(f"{path}: no JSON objects found")
    return objs

def _extract_vertices_edges_from_adj(adj):
    V = set(adj.keys())
    for chs in adj.values():
        if isinstance(chs, list):
            V.update(chs)
    E = []
    for u, chs in adj.items():
        if isinstance(chs, list):
            for v in chs:
                E.append((str(u), str(v)))
    V = sorted(map(str, V), key=lambda x: (len(x), x))
    E = sorted(E, key=lambda e: (e[0], e[1]))
    return V, E

def _normalize_adj_remove_synthetic_root(adj: dict) -> dict:
    """Drop a synthetic 'root' node (if present) from adjacency for building F."""
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if not isinstance(ch, list) or len(ch) != 1:
            raise ValueError("Synthetic 'root' must have exactly one child")
        del adj2["root"]
    return adj2

def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set) -> frozenset:
    """
    Recursively resolve an id to a frozenset of base (negative-string) types.
    - negative id: returns {id}
    - list value: union of resolves
    - single value: resolve that
    Detects cycles and missing entries.
    """
    id_str = str(id_str)
    if id_str.startswith("-"):
        return frozenset([id_str])
    if id_str in memo:
        return memo[id_str]
    if id_str in visiting:
        raise ValueError(f"Cycle detected while resolving potency '{id_str}'")
    if id_str not in comp_map:
        raise ValueError(f"Positive id '{id_str}' appears but not defined in composition map")
    visiting.add(id_str)
    val = comp_map[id_str]
    acc = set()
    if isinstance(val, list):
        for child in val:
            acc |= _resolve_id_to_set(str(child), comp_map, memo, visiting)
    else:
        acc |= _resolve_id_to_set(str(val), comp_map, memo, visiting)
    visiting.remove(id_str)
    memo[id_str] = frozenset(acc)
    return memo[id_str]

def _build_ZA_from_txt(adj: dict, comp_map: dict, unit_drop_edges: bool):
    """
    Build F = (Z_active, A) from adjacency + hierarchical composition map.
    Returns: Z_active, A, base_types(list), potency_id_to_set(dict id->frozenset)
    """
    # Drop synthetic "root" from adjacency for structure building
    adj = _normalize_adj_remove_synthetic_root(adj)
    # Collect all ids we need to resolve
    ids_seen = set(map(str, comp_map.keys()))
    for u, chs in adj.items():
        ids_seen.add(str(u))
        if isinstance(chs, list):
            for v in chs:
                ids_seen.add(str(v))
    memo = {}
    potency_id_to_set = {}
    base_types = set()
    # Resolve every id
    for idv in ids_seen:
        if idv.startswith("-"):
            memo[idv] = frozenset([idv])
        else:
            s = _resolve_id_to_set(idv, comp_map, memo, visiting=set())
            potency_id_to_set[idv] = s
    # Gather base types
    for s in memo.values():
        for t in s:
            if t.startswith("-"):
                base_types.add(t)
    # Z: singletons for all base types + multi-type potencies (size >=2)
    Z_active = {frozenset([t]) for t in base_types}
    for pid, s in potency_id_to_set.items():
        if len(s) >= 2:
            Z_active.add(s)
    # A: only edges in adjacency, mapped via expansion; keep admissible ones
    A = {}
    def id_to_set(x: str) -> frozenset:
        x = str(x)
        if x.startswith("-"):
            return frozenset([x])
        return potency_id_to_set[x]  # safe after resolution above
    for u, chs in adj.items():
        Pu = id_to_set(u)
        for v in chs:
            Qv = id_to_set(v)
            if admissible_edge(Pu, Qv, unit_drop_edges):
                A[(Pu, Qv)] = 1
    return Z_active, A, sorted(base_types), potency_id_to_set

def score_given_map_and_trees(txt_path: str, trees, meta_paths, fixed_k,
                              unit_drop_edges = False):
    """
    Parses the input file and builds the structure F=(Z,A),
    then scores the log-likelihood of the given trees.
    Returns:
        potency_sets (set of frozenset): all potency states
        total_ll (float): total log-likelihood across trees
    """
    objs = _read_json_objects_exact(txt_path)
    if len(objs) < 4:
        raise ValueError("Expected at least 4 JSON lines (adjacency, weights, composition map, root).")

    # 1) adjacency
    adj = None
    for o in objs:
        if isinstance(o, dict) and any(isinstance(v, list) for v in o.values()):
            adj = {str(k): [str(x) for x in v] for k, v in o.items() if isinstance(v, list)}
            break
    if adj is None:
        raise ValueError("Could not locate adjacency dict in the file.")

    # 2) composition map
    comp_map = objs[2]
    if not isinstance(comp_map, dict):
        raise ValueError("Third JSON must be the composition map (dict).")

    # 3) root id
    root_id = objs[3]
    if isinstance(root_id, dict) and "root_id" in root_id:
        root_id = root_id["root_id"]
    root_id = str(root_id)

    # Print vertices and edges
    V, E = _extract_vertices_edges_from_adj(adj)
    # print("=== Parsed Graph: Vertices ===")
    # for v in V: 
    #     print(" ", v)
    # print("\n=== Parsed Graph: Edges (u -> v) ===")
    # for u, v in E: 
    #     print(f"  {u} -> {v}")

    # Build Z, A, and potency definitions
    Z_from_map, A_from_map, base_types_map, potency_def = _build_ZA_from_txt(
        adj=adj,
        comp_map=comp_map,
        unit_drop_edges=unit_drop_edges
    )

    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]
    base_types_data = sorted({str(t) for m in leaf_type_maps for t in m.values()})

    # Merge sets for structure
    S_all = sorted(set(base_types_map) | set(base_types_data))
    Z_active = set(Z_from_map) | {frozenset([t]) for t in S_all}
    A = dict(A_from_map)

    struct = Structure(S=S_all, Z_active=Z_active, A=A, unit_drop=unit_drop_edges)
    dummy_priors = Priors(potency_mode="fixed_k", fixed_k = fixed_k, rho=0.2)

    log_post, per_tree_logs = score_structure(
        struct=struct,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=dummy_priors,
        prune_eps=0.0
    )

    total_ll = sum(per_tree_logs)

    print("\n=== Ground Truth Log-likelihoods (given F from map) ===")
    for i, lg in enumerate(per_tree_logs, 1):
        print(f"Tree {i}: log P(T|F) = {lg:.6f}")
    print(f"Total log-likelihood = {total_ll:.6f}")

    # Convert potency_def dict to set of frozensets
    potency_sets = {frozenset(members) for members in potency_def.values()}
    gt_Z_active = set(Z_active)   # already includes singletons for S_all
    gt_edges = edges_from_A(A)

    return potency_sets, total_ll, gt_Z_active, gt_edges

def jaccard_distance(set1, set2):
    if not set1 and not set2:
        return 0.0
    return 1 - len(set1 & set2) / len(set1 | set2)


def pretty_print_sets(name, sets):
    print(f"\n{name}:")
    for s in sorted(sets, key=lambda x: (len(x), sorted(x))):
        print("  ", sorted(list(s)))


def build_fate_map_path(map_idx: int, type_num: int, tree_kind: str = "graph") -> Tuple[str, str]:
    assert tree_kind in {"graph", "poly_tree", "bin_tree"}
    idx4 = f"{map_idx:04d}"
    fate_map_path = os.path.join(
        "inputs", "differentiation_maps", tree_kind, f"type_{type_num}",
        f"graph_fate_map{idx4}.txt"
    )
    if not os.path.exists(fate_map_path):
        raise FileNotFoundError(f"[fate-map] Not found: {fate_map_path}")
    print(f"[paths] fate_map_path = {fate_map_path}")
    return fate_map_path, idx4


def build_tree_and_meta_paths(map_idx: int, type_num: int, cells_n: int, tree_kind: str = "graph") -> Tuple[List[str], List[str]]:
    assert tree_kind in {"graph", "poly_tree", "bin_tree"}
    idx4 = f"{map_idx:04d}"
    folder = os.path.join("inputs", "trees", tree_kind, f"type_{type_num}", f"cells_{cells_n}")
    tree_paths = [os.path.join(folder, f"{idx4}_tree_{i}.txt") for i in range(5)]
    meta_paths = [os.path.join(folder, f"{idx4}_meta_{i}.txt") for i in range(5)]
    missing = [p for p in (tree_paths + meta_paths) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("[trees/meta] Missing files:\n  " + "\n  ".join(missing))
    print("[paths] tree_paths:");  [print("   ", p) for p in tree_paths]
    print("[paths] meta_paths:");  [print("   ", p) for p in meta_paths]
    return tree_paths, meta_paths


def read_trees_and_maps(tree_paths: List[str], meta_paths: List[str]):
    """
    Load Newick trees and their leaf→type mappings.
    Returns:
        trees          : list[TreeNode]
        leaf_type_maps : list[dict]
        S              : sorted list of all types observed across maps
    """
    for p in tree_paths + meta_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    trees = [read_newick_file(p) for p in tree_paths]
    # raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    raw_maps = [read_leaf_type_map_tls(p) for p in meta_paths]

    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]

    # Build type universe S from whatever is actually present
    S = sorted({str(t) for m in leaf_type_maps for t in m.values()})
    return trees, leaf_type_maps, S


# --- SMALL UTILITIES ---

def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

def pretty_print_sets(name, sets):
    print(f"\n{name}:")
    for s in sorted(sets, key=lambda x: (len(x), sorted(x))):
        print("  ", sorted(list(s)))

def _collect_potencies_in_tree(root: TreeNode) -> Set[frozenset]:
    """Collect all potencies assigned by Fitch union labeling on this tree."""
    pots = set()
    stack = [root]
    while stack:
        v = stack.pop()
        # assign_union_potency has already set v.potency by now
        if hasattr(v, "potency") and v.potency is not None:
            pots.add(frozenset(v.potency))
        for c in v.children:
            stack.append(c)
    return pots

def compute_fitch_potency_probs(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
) -> List[Tuple[frozenset, float]]:
    """
    Runs union-Fitch on each tree, aggregates normalized transition mass per parent potency
    (same as 'row_sum' in init_progenitors_union_fitch), then returns a list of
    (potency_set, probability) sorted by descending probability.

    Probability is defined as:
        prob(P) = row_sum[P] / sum_P row_sum[P]
    For potencies that appeared in Fitch labelings but never as a 'source' of a transition,
    the probability is 0.0.
    """
    ROOT = frozenset(S)
    row_sum: Dict[frozenset, float] = defaultdict(float)
    observed_pots: Set[frozenset] = set()

    for tree, ltm in zip(trees, leaf_type_maps):
        # assign Fitch potencies
        assign_union_potency(tree, ltm)
        # collect all potencies seen on this tree
        observed_pots |= _collect_potencies_in_tree(tree)

        # count transitions and normalize per tree (real transitions only)
        C_T = per_tree_transition_counts(tree)
        T = sum(C_T.values())
        if T == 0:
            continue
        for (i_set, _j_set), cnt in C_T.items():
            incr = cnt / T
            row_sum[i_set] += incr

    total = sum(row_sum.values())
    # include root + all singletons + any observed potencies so the list is complete
    all_to_report = set(observed_pots) | {ROOT} | {frozenset([t]) for t in S}

    if total <= 0:
        # degenerate case: no transitions; assign 0.0 to all reported potencies
        probs = [(P, 0.0) for P in all_to_report]
    else:
        probs = [(P, row_sum.get(P, 0.0) / total) for P in all_to_report]

    # sort by: probability desc, then size desc, then lexicographic
    probs.sort(key=lambda x: (-x[1], -len(x[0]), tuple(sorted(x[0]))))
    return probs

def print_fitch_potency_probs_once(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
    header: str = "",
):
    probs = compute_fitch_potency_probs(S, trees, leaf_type_maps)
    if header:
        print(header)
    print("=== Fitch Potency Probabilities (descending) ===")
    for P, p in probs:
        label = "{" + ",".join(sorted(P)) + "}"
        print(f"  {label:<30}  {p:.6f}")
    print("===============================================")

def check_convergence(all_chain_stats: List[Dict], rhat_threshold: float = 1.05):
    """
    Calculates the R-hat statistic across multiple chains to check for convergence.

    Args:
        all_chain_stats: A list of the 'stats' dictionaries returned from each chain.
        rhat_threshold: The value below which we consider the chains converged.
    """
    print("\n--- Convergence Diagnostics ---")
    if not all_chain_stats or len(all_chain_stats) < 2:
        print("[Warning] Cannot check convergence with fewer than 2 chains.")
        return

    # Extract the log posterior score trace from each chain's stats
    score_traces = [stats.get('scores', []) for stats in all_chain_stats]

    # Ensure the traces are valid for comparison
    if not all(score_traces) or any(len(t) != len(score_traces[0]) for t in score_traces):
        print("[Warning] Chains have different numbers of samples. Cannot compute R-hat.")
        return

    # ArviZ expects data in the format: (n_chains, n_samples)
    posterior_scores = np.array(score_traces)

    # Calculate the R-hat value
    rhat_value = az.rhat(posterior_scores)

    print(f"R-hat statistic on log posterior scores: {rhat_value.item():.4f}")
    if rhat_value.item() < rhat_threshold:
        print(f"[Success] R-hat is below {rhat_threshold}. The chains appear to have converged. ✅")
    else:
        print(f"[Warning] R-hat is above {rhat_threshold}. The chains have NOT converged. ❌")
        print("          Consider increasing the number of steps and/or the burn-in period.")
    print("-----------------------------")

def plot_mcmc_traces(
    all_stats: List[Dict],
    # burn_in: int,
    # thin: int,
    title: str,
    output_path: str
):
    """
    Plots the log posterior score traces from multiple MCMC chains.

    Args:
        all_stats: List of stats dictionaries from each parallel chain.
        burn_in: The number of burn-in iterations.
        thin: The thinning interval.
        title: The title for the plot.
        output_path: The file path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, stats in enumerate(all_stats):
        full_scores = stats.get('all_scores_trace', [])
        if not full_scores:
            print(f"Warning: Chain {i+1} missing 'all_scores_trace'. Skipping.")
            continue
        # scores = stats.get('scores', [])
        # if not scores:
        #     continue

        iterations = range(len(full_scores))
        
        # Calculate the actual iteration numbers for the x-axis
        # iterations = range(burn_in, burn_in + len(scores) * thin, thin)
        ax.plot(iterations, full_scores, label=f'Chain {i+1}', alpha=0.8)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("MCMC Iteration", fontsize=12)
    ax.set_ylabel("Log Posterior Score", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"📈 Plot saved to {output_path}")

def calculate_metrics_and_plots(
    trees: List[TreeNode],
    struct: Structure, # Your bestF_final
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    leaf_type_maps: List[Dict[str, str]],
    output_prefix: str = "metrics"
):
    """
    Calculates Validation Metrics (Transition Matrix, Entropy, Support).
    Generates both CSV reports and Matplotlib Plots (Heatmap, Bar Charts).
    """
    print("\n=== Calculating Validation Metrics & Plots ===")
    
    # --- 1. Re-run Viterbi to get specific labelings (L_MAP) ---
    # We need the exact path assignments to count transitions and support.
    all_L_MAPs = []
    for tree, B_sets in zip(trees, all_B_sets):
        # Note: This uses the FULL Viterbi (with backpointers), not the fast scoring one
        root_table, memo = dp_tree_root_viterbi(
            tree, struct.labels_list, struct.Reach, B_sets
        )
        L_MAP = find_best_viterbi_labeling(tree, root_table, memo)
        all_L_MAPs.append(L_MAP)

    # --- 2. Prepare Data Structures ---
    
    # Identify Progenitors (multi-type) and Terminals (single-type from S)
    # Sort them to ensure the matrix and plots are always in the same order
    progenitors = sorted([P for P in struct.Z_active if len(P) >= 2], 
                         key=lambda x: (len(x), tuple(sorted(list(x)))))
    terminals = sorted(struct.S)
    
    # Init Counts for Transitions: Dict[Progenitor, Dict[Terminal, Count]]
    transition_counts = {P: {t: 0 for t in terminals} for P in progenitors}
    
    # Init Counts for Support
    support_counts = {P: 0 for P in progenitors}
    total_support_C = 0

    # --- 3. Iterate Trees to Accumulate Counts ---
    
    for i, (tree, L_MAP, ltm, B_sets) in enumerate(zip(trees, all_L_MAPs, leaf_type_maps, all_B_sets)):
        if not L_MAP: continue
        
        # A. Count Transitions (Progenitor -> Leaf)
        for u in iter_edges(tree):
            parent, child = u
            
            # Skip if nodes weren't labeled (e.g. due to filtering)
            if parent not in L_MAP or child not in L_MAP: continue
            
            label_P = L_MAP[parent]
            
            # We care about edges where Parent is a Progenitor and Child is a Leaf
            if len(label_P) >= 2 and child.is_leaf():
                leaf_type = ltm.get(child.name)
                
                # Ensure the leaf type is one of our tracked terminals
                if leaf_type and leaf_type in terminals:
                    if label_P in transition_counts:
                        transition_counts[label_P][leaf_type] += 1

        # B. Count Support (Exact Matches of Potency)
        # Paper definition: "number of internal nodes... with exact potency"
        # We check the 'observed' B_set of the node against our inferred Z
        for node, observed_fates in B_sets.items():
            if node.is_leaf(): continue # Skip leaves
            
            obs_froz = frozenset(observed_fates)
            if obs_froz in support_counts:
                support_counts[obs_froz] += 1
                total_support_C += 1

    # --- 4. Organize Data for Plotting ---
    
    # Matrix Data (Rows=Progs, Cols=Terms)
    matrix_data = np.zeros((len(progenitors), len(terminals)), dtype=int)
    prog_labels = [pot_str(P) for P in progenitors]
    term_labels = terminals
    
    # Marginal Data (Total cells derived from each progenitor)
    prog_total_counts = []
    
    for i, P in enumerate(progenitors):
        row_sum = 0
        for j, t in enumerate(terminals):
            cnt = transition_counts[P][t]
            matrix_data[i, j] = cnt
            row_sum += cnt
        prog_total_counts.append(row_sum)

    # Calculate Entropy
    total_cells = sum(prog_total_counts)
    entropy = 0.0
    marginals = []
    if total_cells > 0:
        marginals = [c / total_cells for c in prog_total_counts]
        for p in marginals:
            if p > 0: entropy -= p * math.log(p)
    else:
        marginals = [0.0] * len(progenitors)
            
    # Support Values list
    support_vals = [support_counts[P] for P in progenitors]


    # --- 5. Generate CSV Reports ---
    
    # CSV 1: Transition Matrix
    csv_path_trans = f"{output_prefix}_transition_matrix.csv"
    with open(csv_path_trans, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Progenitor"] + terminals + ["Total_Cells"])
        for i, P in enumerate(progenitors):
            row = [pot_str(P)] + list(matrix_data[i]) + [prog_total_counts[i]]
            writer.writerow(row)
    print(f"Saved Transition Matrix CSV: {csv_path_trans}")

    # CSV 2: Marginals & Entropy
    csv_path_marg = f"{output_prefix}_marginal_entropy.csv"
    with open(csv_path_marg, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Cells", total_cells])
        writer.writerow(["Entropy (H)", f"{entropy:.4f}"])
        writer.writerow([])
        writer.writerow(["Progenitor", "Count", "Proportion"])
        for i, P in enumerate(progenitors):
             writer.writerow([pot_str(P), prog_total_counts[i], f"{marginals[i]:.4f}"])
    print(f"Saved Entropy CSV (H={entropy:.4f}): {csv_path_marg}")

    # CSV 3: Support
    csv_path_supp = f"{output_prefix}_support.csv"
    with open(csv_path_supp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Support (C)", total_support_C])
        writer.writerow([])
        writer.writerow(["Progenitor", "Support_Count"])
        for i, P in enumerate(progenitors):
             writer.writerow([pot_str(P), support_vals[i]])
    print(f"Saved Support CSV (C={total_support_C}): {csv_path_supp}")


    # --- 6. Generate Plots (Matplotlib) ---
    
    # Plot 1: Transition Matrix Heatmap (Fig 3f)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im = ax1.imshow(matrix_data, cmap='OrRd', aspect='auto')
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(progenitors)):
        for j in range(len(terminals)):
            c = matrix_data[i, j]
            # Choose text color based on background intensity
            text_color = "white" if c > matrix_data.max() / 2 else "black"
            ax1.text(j, i, str(c), ha="center", va="center", color=text_color, fontsize=9)

    ax1.set_xticks(np.arange(len(terminals)))
    ax1.set_yticks(np.arange(len(progenitors)))
    ax1.set_xticklabels(term_labels, rotation=45, ha="right")
    ax1.set_yticklabels(prog_labels)
    ax1.set_xlabel("Observed Cell Types (Terminals)")
    ax1.set_ylabel("Inferred Progenitors")
    ax1.set_title(f"Transition Counts (Total Cells: {total_cells})")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_transitions.png", dpi=300)
    plt.close(fig1)
    print(f"Saved Plot: {output_prefix}_transitions.png")

    # Plot 2: Marginal Distribution (Fig 3g)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(progenitors))
    ax2.bar(x_pos, marginals, color='black', width=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(prog_labels, rotation=45, ha="right")
    ax2.set_ylabel("Proportion of Cells")
    ax2.set_title(f"Marginal Distribution (Entropy H = {entropy:.3f})")
    ax2.set_ylim(0, 1.0) # Scale 0 to 1 like the paper
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_marginal.png", dpi=300)
    plt.close(fig2)
    print(f"Saved Plot: {output_prefix}_marginal.png")

    # Plot 3: Support (Fig 3h)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(x_pos, support_vals, color='black', width=0.6)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(prog_labels, rotation=45, ha="right")
    ax3.set_ylabel("Support (Number of Internal Nodes)")
    ax3.set_title(f"Progenitor Support (Total C = {total_support_C})")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_support.png", dpi=300)
    plt.close(fig3)
    print(f"Saved Plot: {output_prefix}_support.png")

    return entropy, total_support_C


def plot_final_structure(struct: Structure, output_path: str):
    """
    Generates a hierarchical DAG plot of the inferred structure using NetworkX and Matplotlib.
    Nodes are colored by potency size (cardinality).
    """
    print(f"\n=== Generating Structure Plot: {output_path} ===")
    
    # 1. Create Graph
    G = nx.DiGraph()
    
    # Add Nodes (frozensets)
    G.add_nodes_from(struct.Z_active)
    
    # Add Edges (only where A[e] == 1)
    edges = [e for e, v in struct.A.items() if v == 1]
    G.add_edges_from(edges)
    
    # 2. Prepare Attributes (Labels and Counts)
    node_counts_map = {n: len(n) for n in G.nodes()}
    
    # Format labels: remove brackets, split elements by newlines for compactness
    node_labels_map = {}
    for n in G.nodes():
        elements = sorted(list(n))
        node_labels_map[n] = '\n'.join(elements)

    # 3. Hierarchical Layout (y = cardinality)
    pos = {}
    levels = {}
    for node, count in node_counts_map.items():
        levels.setdefault(count, []).append(node)

    x_spacing = 2.0
    y_spacing = 2.0

    sorted_counts = sorted(levels.keys())
    for count in sorted_counts:
        nodes_at_level = sorted(levels[count], key=lambda x: tuple(sorted(list(x))))
        num_nodes = len(nodes_at_level)
        x_start = - (num_nodes - 1) * x_spacing / 2.0
        for i, node in enumerate(nodes_at_level):
            y = count * y_spacing
            x = x_start + i * x_spacing
            pos[node] = (x, y)

    # 4. Plotting Setup
    if not node_counts_map: 
        print("Warning: Empty graph, skipping plot.")
        return

    color_values = [node_counts_map[node] for node in G.nodes()]
    max_potency = max(color_values)
    min_potency = min(color_values)
    num_levels = max(1, max_potency - min_potency + 1)
    
    # Create colormap
    cmap = plt.get_cmap('YlOrRd', num_levels)

    plt.figure(figsize=(20, 16))
    
    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=color_values,
        cmap=cmap,
        vmin=min_potency,
        vmax=max_potency,
        node_size=8000,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Draw Edges
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=25,
        edge_color='dimgrey',
        width=1.5,
        node_size=8000
    )

    # 5. Draw Labels
    # Identify topmost nodes for white text (usually darker background color)
    max_count = max(node_counts_map.values())
    top_nodes = [n for n, c in node_counts_map.items() if c == max_count]

    # Draw all labels in black first
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels_map,
        font_size=8,
        font_weight='bold',
        font_color='black'
    )

    # Overdraw labels for topmost node(s) in white for contrast
    for node in top_nodes:
        nx.draw_networkx_labels(
            G, pos,
            labels={node: node_labels_map[node]},
            font_size=8,
            font_weight='bold',
            font_color='white'
        )

    # 6. Colorbar
    boundaries = np.arange(min_potency, max_potency + 2) - 0.5
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(color_values)
    ticks = np.arange(min_potency, max_potency + 1)
    
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, ticks=ticks)
    cbar.set_label('Cardinality (Number of Potencies)', rotation=270, labelpad=25, fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.title(f'Inferred differentiation Map', size=25, pad=20, weight='bold')
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Saved plot to {output_path}")


def process_case( # This function REPLACES your old `process_case`
    map_idx: int, 
    type_num: int, 
    cells_n: int,
    priors: "Priors", 
    iters: int, 
    restarts: int,  # This is now n_chains for Phase 1
    log_dir: Optional[str] = None,
    tree_kind: str = "graph", 
    n_jobs: Optional[int] = None,
    unit_drop_edges: bool = False # Make sure this is passed
):
    """
    Processes a single simulation case using the 
    "MCMC-Z -> Viterbi Flow -> MST" hybrid algorithm.
    """
    
    # --- 0. Setup ---
    print(f"\n{'='*80}")
    print(f"=== Processing Case: type_{type_num}, map {map_idx:04d}, cells_{cells_n} ===")
    print(f"=== Sampler: MCMC-Z (Phase 1) + Viterbi Flow (Phase 2) ===")
    print(f"=== k={priors.fixed_k}, iters={iters}, chains={restarts} ===")
    print(f"{'='*80}")

    fate_map_path, idx4 = build_fate_map_path(map_idx, type_num, tree_kind=tree_kind)
    tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n, tree_kind=tree_kind)

    trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
    all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

    # --- Ground Truth Scoring (Unchanged) ---
    ground_truth_sets, gt_loss, gt_Z_active, gt_edges = score_given_map_and_trees(
        fate_map_path, trees, meta_paths,
        fixed_k=priors.fixed_k,
        unit_drop_edges=unit_drop_edges
    )
    
    # --- 1. Prepare for MCMC (Unchanged) ---
    print_fitch_potency_probs_once(
        S, trees, leaf_type_maps,
        header=f"\n[Potency ranking] type_{type_num}, map {idx4}, cells_{cells_n}"
    )
    pool = collect_fitch_multis(S, trees, leaf_type_maps)
    fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
    fitch_probs_dict = {p: prob for p, prob in fitch_probs_list}

    # --- 2. Phase 1: Run MCMC for Z_map ---
    # This uses your *existing* Z-only MCMC function
    print("\n--- Phase 1: Running MCMC to find best Potency Sets (Z) ---")
    bestF_Z, best_score_Z, all_stats_Z = run_mcmc_only_Z_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        # all_B_sets=all_B_sets,
        priors=priors,
        unit_drop_edges=unit_drop_edges, # Pass this along
        fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
        steps=iters,
        burn_in=(iters*15)//100,
        thin=10,
        base_seed=123 + map_idx + cells_n,
        candidate_pool=pool,
        block_swap_sizes=(1,), # Use (1,) for faster, simpler swaps
        n_chains=restarts,
        fitch_probs = fitch_probs_dict
    )

    if all_stats_Z and log_dir:
        plot_mcmc_traces(
            all_stats=all_stats_Z,
            title=f"MCMC Trace for Potency Sets (Z) - Map {idx4}, Cells {cells_n}",
            output_path=os.path.join(log_dir, f"trace_Z_type{type_num}_{idx4}_cells{cells_n}.png")
        )

    if bestF_Z is None:
        print("[ERROR] MCMC-Z Phase 1 failed to find a structure.")
        raise RuntimeError("MCMC-Z failed")
        
    Z_map = bestF_Z.Z_active
    print(f"--- Phase 1 Complete. Best Z-only score: {best_score_Z:.4f} ---")

    # --- 3. Phase 2: Create F_full ---
    print("\n--- Phase 2: Building fully-connected graph (F_full) ---")
    A_full = _full_edges_for_Z(Z_map, unit_drop_edges)
    F_full = Structure(S, Z_map, A_full, unit_drop_edges)
    print(f"F_full has {len(Z_map)} nodes and {len(A_full)} admissible edges.")

    # --- 4. Phase 3: Calculate Viterbi Flow ---
    viterbi_flow = calculate_viterbi_flow(
        trees, F_full, all_B_sets, leaf_type_maps
    )
    
    if not viterbi_flow:
        print("[ERROR] Viterbi flow calculation returned empty. Cannot build A_map.")
        raise RuntimeError("Viterbi flow failed")

    # --- 5. Phase 4: Select A_map ---
    S_nodes = {frozenset([t]) for t in S} # Get the set of singletons
    A_map_final = build_A_map_from_flow(
        viterbi_flow, 
        Z_map,
        S_nodes, # <-- ADD THIS ARGUMENT
        z_score_threshold=1.5 # You can tune this
    )
    
    # --- 6. Final Scoring and Reporting ---
    print("\n--- Final Scoring ---")
    bestF_final = Structure(S, Z_map, A_map_final, unit_drop_edges)
    
    # Score this final structure
    # final_logp_Z = priors.log_prior_Z(S, Z_map)
    # final_num_admiss = len(A_full) # Num admissible edges for this Z
    # final_logp_A = priors.log_prior_A(
    #     Z_map, A_map_final, unit_drop_edges, final_num_admiss
    # )

    # final_log_L = get_log_likelihood(
    #     bestF_final, trees, leaf_type_maps, all_B_sets
    # )
    
    # if not math.isfinite(final_logp_Z): final_logp_Z = 0 # Handle -inf prior
    # if not math.isfinite(final_logp_A): final_logp_A = 0 # Handle -inf prior
    
    # best_score_final = final_logp_Z + final_logp_A + final_log_L

    best_score_final, _ = score_structure(bestF_final, trees, leaf_type_maps, priors)

    # print(f"Final Log P(Z): {final_logp_Z:.4f}")
    # print(f"Final Log P(A|Z): {final_logp_A:.4f} ({len(A_map_final)} edges)")
    print(f"Final score (with edges): {best_score_final:.4f}")
    
    # print(f"\n=== BEST MAP (Hybrid) for type_{type_num}, map {idx4}, cells_{cells_n} ===")
    # multi_sorted = sorted([P for P in bestF_final.Z_active if len(P) >= 2],
    #                       key=lambda x: (len(x), tuple(sorted(list(x)))))
    # print("Active potencies (multi-type):")
    # for P in multi_sorted: print("  ", pot_str(P))
    
    # print("\nEdges:")
    # edges = sorted([e for e, v in bestF_final.A.items() if v == 1],
    #                key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1])))
    # for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  Log Posterior (Pred): {best_score_final:.6f}")
    print(f"  Log Posterior (GT):   {gt_loss:.6f}")

    # --- Potency Set Metrics ---
    predicted_sets = {p for p in bestF_final.Z_active if len(p) > 1}
    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", ground_truth_sets)
    print("\nPredicted edges")
    edges = sorted([e for e, v in bestF_final.A.items() if v == 1],
                   key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1])))
    for (u, v) in sorted(edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
        print(f"{sorted(list(u))} -> {sorted(list(v))}")
    # for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")
    print("\n=== Ground Truth Directed Edges ===")
    for (u, v) in sorted(gt_edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
        print(f"{sorted(list(u))} -> {sorted(list(v))}")
    jd = jaccard_distance(predicted_sets, ground_truth_sets)
    print(f"Jaccard Distance (Potency Sets): {jd:.6f}")

    # --- Edge Set Metrics ---
    pred_edges = edges_from_A(bestF_final.A)
    edge_jacc = jaccard_distance_edges(pred_edges, gt_edges)
    
    nodes_union = gt_Z_active | set(bestF_final.Z_active)
    im_d, im_s = ipsen_mikhailov_similarity(
        nodes_union = nodes_union,
        edges1 = pred_edges,
        edges2 = gt_edges,
        gamma = 0.08,
    )

    print("\n=== Edge-set Metrics ===")
    print(f"Jaccard distance (edges): {edge_jacc:.6f}")
    print(f"Ipsen–Mikhailov distance: {im_d:.6f}")
    print(f"Ipsen–Mikhailov similarity: {im_s:.6f}")

    return jd, gt_loss, best_score_final, edge_jacc, im_s

def main_tls(
    input_file_list: str, # Path to the file listing tree/meta pairs
    iters: int = 500,     # MCMC iterations (consider increasing)
    restarts: int = 4,    # Number of parallel chains
    num_cores: int = 7,
    fixed_k: int = 10,    # Number of multi-type potencies
    out_csv: str = "tls_results.csv",
    log_dir: Optional[str] = "tls_logs",
    # unit_drop_edges: bool = False # Set based on your model assumption
):
    random.seed(42) # Or another seed
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
    unit_drop_edges = False # Usually False for biological realism unless specified

    try:
        # 1. Parse the input file list to get paths
        tree_paths, meta_paths = parse_input_file_list(input_file_list)

        # 2. Load all trees and maps
        trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
        print(f"Loaded {len(trees)} trees. Type universe S: {S}")

        # 3. Precompute B_sets for all trees
        all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

        # 4. (Optional but recommended) Get candidate pool and Fitch probs
        print_fitch_potency_probs_once(
            S, trees, leaf_type_maps,
            header=f"\n[Potency ranking] from {input_file_list}"
        )
        pool = collect_fitch_multis(S, trees, leaf_type_maps)
        fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
        fitch_probs_dict = {p: prob for p, prob in fitch_probs_list}


        # --- MCMC Phase 1: Search for Best Z ---
        print("\n--- Starting MCMC Phase 1: Searching for Potency Sets (Z) ---")
        bestF_Z, best_score_Z, all_stats_Z = run_mcmc_only_Z_parallel(
            S=S,
            trees=trees,
            leaf_type_maps=leaf_type_maps,
            # all_B_sets=all_B_sets,
            priors=priors,
            unit_drop_edges=unit_drop_edges, # Use consistent setting
            fixed_k=fixed_k,
            steps=iters,
            burn_in=(iters * 15) // 100,
            thin=10,
            base_seed=123,
            candidate_pool=pool,
            block_swap_sizes=(1,), # Usually swap 1 at a time is fine
            n_chains=restarts,
            num_cores = num_cores,
            fitch_probs=fitch_probs_dict
        )

        if all_stats_Z and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            plot_mcmc_traces(
                all_stats=all_stats_Z,
                title=f"MCMC Trace for Potency Sets (Z)",
                output_path=os.path.join(log_dir, f"trace_Z_tls.png")
            )

        if bestF_Z is None:
            print("[ERROR] MCMC-Z Phase 1 failed to find a structure.")
            raise RuntimeError("MCMC-Z failed")

        Z_map = bestF_Z.Z_active
        print(f"--- MCMC Phase 1 Complete. Best Z score: {best_score_Z:.4f} ---")
        # check_convergence(all_stats_Z) # Check convergence for Z search

        # --- 3. Phase 2: Create F_full ---
        print("\n--- Phase 2: Building fully-connected graph (F_full) ---")
        A_full = _full_edges_for_Z(Z_map, unit_drop_edges)
        F_full = Structure(S, Z_map, A_full, unit_drop_edges)
        print(f"F_full has {len(Z_map)} nodes and {len(A_full)} admissible edges.")

        # --- 4. Phase 3: Calculate Viterbi Flow ---
        viterbi_flow = calculate_viterbi_flow(
            trees, F_full, all_B_sets, leaf_type_maps, num_cores=num_cores
        )
        
        if not viterbi_flow:
            print("[ERROR] Viterbi flow calculation returned empty. Cannot build A_map.")
            raise RuntimeError("Viterbi flow failed")

        # --- 5. Phase 4: Select A_map ---
        S_nodes = {frozenset([t]) for t in S} # Get the set of singletons
        A_map_final = build_A_map_from_flow(
            viterbi_flow, 
            Z_map,
            S_nodes, # <-- ADD THIS ARGUMENT
            z_score_threshold=1.5 # You can tune this
        )
        
        # --- 6. Final Scoring and Reporting ---
        print("\n--- Final Scoring ---")
        bestF_final = Structure(S, Z_map, A_map_final, unit_drop_edges)
        
        # Score this final structure
        # final_logp_Z = priors.log_prior_Z(S, Z_map)
        # final_num_admiss = len(A_full) # Num admissible edges for this Z
        # final_logp_A = priors.log_prior_A(
        #     Z_map, A_map_final, unit_drop_edges, final_num_admiss
        # )

        # final_log_L = get_log_likelihood(
        #     bestF_final, trees, leaf_type_maps, all_B_sets
        # )
        
        # if not math.isfinite(final_logp_Z): final_logp_Z = 0 # Handle -inf prior
        # if not math.isfinite(final_logp_A): final_logp_A = 0 # Handle -inf prior
        
        # best_score_final = final_logp_Z + final_logp_A + final_log_L

        best_score_final, _ = score_structure_parallel(
            bestF_final, trees, leaf_type_maps, priors, num_cores=num_cores
        )

        best_greedy_score = score_structure_greedy(bestF_final, trees, leaf_type_maps, priors)

        # print(f"Final Log P(Z): {final_logp_Z:.4f}")
        # print(f"Final Log P(A|Z): {final_logp_A:.4f} ({len(A_map_final)} edges)")
        print("\nScores:")
        print(f"Final score (with edges): {best_score_final:.4f}")
        print(f"Greedy score (with edges): {best_greedy_score:.4f}")

        
        # print(f"\n=== BEST MAP (Hybrid) for type_{type_num}, map {idx4}, cells_{cells_n} ===")
        # multi_sorted = sorted([P for P in bestF_final.Z_active if len(P) >= 2],
        #                       key=lambda x: (len(x), tuple(sorted(list(x)))))
        # print("Active potencies (multi-type):")
        # for P in multi_sorted: print("  ", pot_str(P))
        
        # print("\nEdges:")
        # edges = sorted([e for e, v in bestF_final.A.items() if v == 1],
        #                key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1])))
        # for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

        
        # print(f"  Log Posterior (GT):   {gt_loss:.6f}")

        # --- Potency Set Metrics ---
        predicted_sets = {p for p in bestF_final.Z_active if len(p) > 1}
        pretty_print_sets("Predicted Sets", predicted_sets)
        # pretty_print_sets("Ground Truth Sets", ground_truth_sets)
        print("\nPredicted edges")
        edges = sorted([e for e, v in bestF_final.A.items() if v == 1],
                    key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1])))
        for (u, v) in sorted(edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
            print(f"{sorted(list(u))} -> {sorted(list(v))}")
        # for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")
        # print("\n=== Ground Truth Directed Edges ===")
        # for (u, v) in sorted(gt_edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
        #     print(f"{sorted(list(u))} -> {sorted(list(v))}")
        # jd = jaccard_distance(predicted_sets, ground_truth_sets)
        # print(f"Jaccard Distance (Potency Sets): {jd:.6f}")

        # --- Edge Set Metrics ---
        # pred_edges = edges_from_A(bestF_final.A)
        # edge_jacc = jaccard_distance_edges(pred_edges, gt_edges)
        
        # nodes_union = gt_Z_active | set(bestF_final.Z_active)
        # im_d, im_s = ipsen_mikhailov_similarity(
        #     nodes_union = nodes_union,
        #     edges1 = pred_edges,
        #     edges2 = gt_edges,
        #     gamma = 0.08,
        # )

        # print("\n=== Edge-set Metrics ===")
        # print(f"Jaccard distance (edges): {edge_jacc:.6f}")
        # print(f"Ipsen–Mikhailov distance: {im_d:.6f}")
        # print(f"Ipsen–Mikhailov similarity: {im_s:.6f}")

        # # --- MCMC Phase 2: Search for Best A given Best Z ---
        # print("\n--- Starting MCMC Phase 2: Searching for Edges (A) given best Z ---")
        # # Increase iterations slightly for edge search maybe
        # iters_A = iters + 100
        # # + max(50, iters // 10) # Example: add 10% or 50 steps

        # bestF_A, best_score_A, all_stats_A = run_mcmc_only_A_parallel(
        #     S=S,
        #     trees=trees,
        #     leaf_type_maps=leaf_type_maps,
        #     all_B_sets=all_B_sets,
        #     priors=priors,
        #     unit_drop_edges=unit_drop_edges, # Use consistent setting
        #     fixed_k=fixed_k,
        #     steps=iters_A,
        #     burn_in=(iters_A * 15) // 100,
        #     thin=10,
        #     base_seed=456, # Use a different base seed
        #     candidate_pool=pool, # Not strictly needed but passed
        #     block_swap_sizes=(1,), # Not used in A-only search
        #     n_chains=restarts,
        #     Z=bestF_Z.Z_active # <<< Pass the best Z found in phase 1
        # )

        # if bestF_A is None:
        #     print("ERROR: MCMC Phase 2 (A search) failed. Using Z-search result.")
        #     bestF = bestF_Z # Fallback
        #     best_score = best_score_Z
        # else:
        #      print(f"--- MCMC Phase 2 Complete. Best (Z,A) score: {best_score_A:.4f} ---")
        #     #  check_convergence(all_stats_A) # Check convergence for A search
        #      bestF = bestF_A
        #      best_score = best_score_A


        # # --- Output Results ---
        # if log_dir:
        #     os.makedirs(log_dir, exist_ok=True)
        #     # Plot traces if needed (using all_stats_Z and all_stats_A)
        #     if all_stats_Z:
        #          plot_mcmc_traces(
        #             all_stats=all_stats_Z, 
        #             # burn_in=(iters * 15)//100, 
        #             # thin=10,
        #             title=f"MCMC Trace (Z search) - {os.path.basename(input_file_list)}",
        #             output_path=os.path.join(log_dir, f"trace_Z_{os.path.basename(input_file_list)}.png")
        #          )
        #     if all_stats_A:
        #          plot_mcmc_traces(
        #             all_stats=all_stats_A, 
        #             # burn_in=(iters_A * 15)//100, 
        #             # thin=10,
        #             title=f"MCMC Trace (A search) - {os.path.basename(input_file_list)}",
        #             output_path=os.path.join(log_dir, f"trace_A_{os.path.basename(input_file_list)}.png")
        #          )


        # print("\n=== FINAL BEST MAP STRUCTURE ===")
        # multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
        #                       key=lambda x: (len(x), tuple(sorted(list(x)))))
        # print("Active potencies (multi-type):")
        # for P in multi_sorted: print("  ", pot_str(P))
        # print("\nSingletons (always active):")
        # for t in S: print("  ", "{" + t + "}")

        # print("\nEdges:")
        # edges = sorted([e for e, v in bestF.A.items() if v == 1],
        #                key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
        # for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

        # print(f"\nFinal Log Posterior Score: {best_score:.6f}")

        # --- Save summary (optional) ---
        entropy, support = calculate_metrics_and_plots(
            trees, 
            bestF_final, 
            all_B_sets, 
            leaf_type_maps,
            output_prefix=os.path.join(log_dir if log_dir else ".", "tls_validation")
        )

        
        print(f"Validation Metrics -> Entropy: {entropy:.4f}, Support: {support}")
        
        
        # --- Generate Structure Plot ---
        # Define output filename
        plot_filename = f"tls_structure.png"
        plot_path = os.path.join(log_dir if log_dir else ".", plot_filename)
        
        # Call the new function
        plot_final_structure(bestF_final, plot_path)

        with open(out_csv, "w", newline="") as f:
             writer = csv.writer(f)
             writer.writerow(["InputFile", "FixedK", "BestLogPosterior", "NumEdges"])
             writer.writerow([
                 input_file_list, fixed_k, f"{best_score_final:.6f}", len(edges)
             ])
        print(f"\nResults summary saved to {out_csv}")



    except FileNotFoundError as e:
        print(f"[ERROR] Input file not found: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main_tls(
         # **CHANGED**: Updated the input file path
        input_file_list="TLS_locations.txt",
        iters=100,
        restarts=7,
        num_cores = 7,
        fixed_k=6,
        out_csv="tls_6_fast.csv",
        log_dir="tls_run_logs_fast_6"
    )