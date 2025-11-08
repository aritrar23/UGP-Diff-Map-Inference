from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import sys
import random
import csv
import json
import math, random
import itertools
from typing import Iterable,  Dict, Tuple, List, Optional, Set, FrozenSet, Any
from collections import Counter, defaultdict
from tqdm import trange, tqdm
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST BE CALLED BEFORE importing pyplot
import matplotlib.pyplot as plt
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


def build_A_map_from_flow(
    viterbi_flow: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float],
    Z_map: Set[FrozenSet[str]],
    S_nodes: Set[FrozenSet[str]], # Set of singletons
    z_score_threshold: float = 1.5 # This argument is no longer used, but kept for compatibility
) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]:
    """
    Builds the final A_map using the:
    1. "Max-Flow Backbone" (MST)
    2. "Branching Fix" (>= 2 children)
    3. "20% Flow Rule" (Paper's heuristic)
    """
    print("Building final A_map from Viterbi flow...")
    A_map: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int] = {}
    
    if not viterbi_flow:
        print("Warning: Viterbi flow is empty. Returning empty edge set.")
        return {}

    # --- 1. Step 4a: Find the "Max-Flow Backbone" (MST) ---
    # (This section is UNCHANGED)
    
    undirected_weights: Dict[FrozenSet[Tuple], float] = defaultdict(float)
    for (P, Q), flow in viterbi_flow.items():
        edge_key = frozenset([P, Q])
        if flow > undirected_weights[edge_key]:
            undirected_weights[edge_key] = flow
    
    kruskal_edge_list = []
    for edge_key, weight in undirected_weights.items():
        if weight > 0:
            P, Q = tuple(edge_key)
            kruskal_edge_list.append((weight, (P, Q)))

    print(f"Finding Maximum Spanning Tree (Backbone) from {len(kruskal_edge_list)} weighted edges...")
    mst_edges_undir = kruskal_mst(Z_map, kruskal_edge_list)
    
    print(f"Backbone (from MST) contains {len(mst_edges_undir)} directed edges:")
    for (P, Q) in mst_edges_undir:
        flow_PQ = viterbi_flow.get((P, Q), 0)
        flow_QP = viterbi_flow.get((Q, P), 0)
        
        edge_to_add = (P, Q) if flow_PQ >= flow_QP else (Q, P)
        A_map[edge_to_add] = 1
        print(f"  [MST] Added: {pot_str(edge_to_add[0])} -> {pot_str(edge_to_add[1])} (w={max(flow_PQ, flow_QP):.2f})")


    # --- 2. Step 4b: Enforce >= 2 Children Constraint ---
    # (This section is UNCHANGED)
    
    print("Enforcing >= 2 children constraint for progenitors...")
    progenitor_nodes = Z_map - S_nodes
    added_for_branching = 0
    
    for P in progenitor_nodes:
        current_out_degree = sum(1 for (parent, child) in A_map if parent == P)
        
        needed = 2 - current_out_degree
        if needed <= 0:
            continue
            
        missing_edges = []
        for Q in Z_map:
            edge = (P, Q)
            if P != Q and edge not in A_map and viterbi_flow.get(edge, 0) > 0:
                missing_edges.append((viterbi_flow[edge], edge))
        
        missing_edges.sort(key=lambda x: x[0], reverse=True)
        
        edges_to_add = missing_edges[:needed]
        
        if edges_to_add:
            print(f"  Fixing {pot_str(P)} (needs {needed} more):")
            for flow, edge in edges_to_add:
                A_map[edge] = 1
                added_for_branching += 1
                print(f"    [BRANCH] Added: {pot_str(edge[0])} -> {pot_str(edge[1])} (w={flow:.2f})")
            
    print(f"Added {added_for_branching} edges to satisfy branching constraint.")

    # --- 3. Step 4c: Add Secondary Paths (Paper's 20% Rule) ---
    # (This section is NEW and replaces the z-score logic)
    
    print("Adding secondary edges with > 20% of parent's total flow...")

    # First, pre-calculate the total *positive* Viterbi flow out of *every* node
    total_flow_out = defaultdict(float)
    for (P, Q), flow in viterbi_flow.items():
        if flow > 0:
            total_flow_out[P] += flow
            
    added_for_flow = 0
    
    # Now, check every potential edge that passed the Viterbi step
    for (P, Q), flow in viterbi_flow.items():
        edge = (P, Q)
        
        # Check 1: Is it a positive-flow edge?
        if flow <= 0:
            continue
            
        # Check 2: Is it *already* in our map? (from MST or branching)
        if edge in A_map:
            continue
            
        # Check 3: Is the parent a progenitor? (We don't care about flow from singletons)
        if P in S_nodes:
            continue
            
        # Check 4: Does it meet the 20% threshold?
        parent_total_flow = total_flow_out.get(P, 0)
        
        # Avoid division by zero, and apply the rule
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


# def parse_newick(newick: str) -> TreeNode:
#     s = newick.strip()
#     if not s.endswith(";"): raise ValueError("Newick must end with ';'")
#     s = s[:-1]; i = 0
#     def parse() -> TreeNode:
#         nonlocal i, s
#         if i >= len(s): raise ValueError("Unexpected end")
#         if s[i] == '(':
#             i += 1
#             node = TreeNode()
#             while True:
#                 node.add_child(parse())
#                 if i >= len(s): raise ValueError("Unbalanced")
#                 if s[i] == ',':
#                     i += 1; continue
#                 elif s[i] == ')':
#                     i += 1; break
#                 else: raise ValueError(f"Unexpected char: {s[i]} at {i}")
#             j = i
#             while j < len(s) and s[j] not in ',()': j += 1
#             name = s[i:j].strip()
#             if name: node.name = name
#             i = j
#             return node
#         else:
#             j = i
#             while j < len(s) and s[j] not in ',()': j += 1
#             name = s[i:j].strip()
#             if not name: raise ValueError("Leaf without name")
#             i = j
#             return TreeNode(name=name)
#     root = parse()
#     if i != len(s): raise ValueError(f"Trailing characters: '{s[i:]}'")
#     return root

class SubsetReach:
    """
    A dummy 'Reach' object that mimics a dictionary for the Z-only MCMC.
    It defines reachability simply as the subset relationship, avoiding the
    need to compute or store the full transitive closure.
    """
    def get(self, key: FrozenSet[str], default: List) -> List[FrozenSet[str]]:
        # In the Z-only search, A is always fully connected, so reachability
        # is just the subset relation.
        # 'key' is the parent potency P. We are looking for all reachable Q's.
        
        # This is a placeholder that we won't actually use. The logic
        # is handled inside the DP function directly now.
        return default

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

#Added during TLS modfications
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

### Pre-TLS function

# def assign_union_potency(root: TreeNode, leaf_type_map: Dict[str, str]) -> Set[str]:
#     """
#     Post-order union-only labeling. Sets `node.potency` for every node (as a Python set).
#     For leaves, looks up leaf_type_map[node.name] to get the leaf cell type.
#     Returns the potency set at `root`.
#     """
#     if root.is_leaf():
#         if root.name is None:
#             raise KeyError("Leaf has no .name; cannot map to leaf_type_map")
#         if root.name not in leaf_type_map:
#             raise KeyError(f"Leaf name '{root.name}' not found in leaf_type_map")
#         root.potency = {leaf_type_map[root.name]}
#         return root.potency

#     union_set: Set[str] = set()
#     for child in root.children:
#         child_set = assign_union_potency(child, leaf_type_map)
#         union_set |= child_set
#     root.potency = union_set
#     return root.potency


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

import os # Make sure os is imported at the top of your file

def create_virtual_star_tree(potency_set: FrozenSet[str], potency_id: str) -> Tuple[TreeNode, Dict[str, str]]:
    """
    Creates an in-memory star-shaped TreeNode and metadata map for a given potency set.

    Args:
        potency_set: A frozenset of cell type strings observed in this potency group.
        potency_id: A unique identifier for this potency group (e.g., row index or 'Potency_X').

    Returns:
        A tuple: (root_node, leaf_type_map)
            - root_node: The TreeNode representing the star tree.
            - leaf_type_map: Metadata dictionary mapping generic leaf names to cell types.
    """
    if not potency_set: # Handle empty potency case if it occurs
        root = TreeNode(name=f"Root_{potency_id}")
        return root, {}

    root = TreeNode(name=f"Root_{potency_id}")
    leaf_type_map = {}

    # Create one leaf node for each unique cell type in the set
    for i, cell_type in enumerate(sorted(list(potency_set))):
        # Create a unique, generic leaf name
        leaf_name = f"{potency_id}_leaf_{cell_type}_{i}"
        leaf_node = TreeNode(name=leaf_name)
        root.add_child(leaf_node)
        leaf_type_map[leaf_name] = cell_type

    return root, leaf_type_map

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

        parent_reach = []

        if isinstance(Reach, SubsetReach):
            # If it's the Z-only search, reachability is just the subset rule.
            # 'P' is the parent label. We find all active labels 'L' that are subsets of P.
            if P is None:
                parent_reach = active_labels
            else:
                parent_reach = [L for L in active_labels if L.issubset(P)]

        else:
            # Otherwise, use the standard dictionary lookup for the A-only search.
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

    # def log_prior_A(self, Z_active:Set[FrozenSet[str]], A:Dict[Tuple[FrozenSet[str],FrozenSet[str]],int], unit_drop=True)->float:
    #     # ------------------------------------------------------------
    #     # Computes log P(A | Z): the log prior over EDGE EXISTENCE between active potencies.
    #     #
    #     # Inputs:
    #     #   - Z_active: set of active potencies (nodes in the potency DAG)
    #     #   - A: adjacency dictionary mapping (P,Q) -> {0,1}, indicating whether edge P->Q is present
    #     #   - unit_drop: if True, an admissible edge must drop EXACTLY one fate (|P\Q| == 1);
    #     #                otherwise any monotone subset drop (Q ⊂ P) is admissible.
    #     #
    #     # Prior:
    #     #   - For every admissible pair (P,Q):
    #     #         A_{P->Q} ~ Bernoulli(rho)
    #     #     So:
    #     #         log P(A|Z) = ∑_{(P,Q) admissible} [ A_{P->Q} log(rho) + (1 - A_{P->Q}) log(1 - rho) ]
    #     #
    #     # Notes:
    #     #   - "Admissible" enforces graph shape constraints (subset-monotone and possibly unit-drop).
    #     #   - If an edge (P,Q) is not admissible, it does not contribute to the product/sum at all.
    #     # ------------------------------------------------------------
    #     labels=list(Z_active)
    #     # admissible set is pairs with subset monotone (and optionally unit-drop)
    #     logp=0.0
    #     for P in labels:
    #         for Q in labels:
    #             if admissible_edge(P,Q,unit_drop):
    #                 # a == 1 if the edge is present in A, else 0
    #                 a = 1 if A.get((P,Q),0)==1 else 0
    #                 # add Bernoulli log-prob for this edge
    #                 logp += math.log(self.rho) if a==1 else math.log(1-self.rho)
    #     return logp

    # --- Replace the existing log_prior_A method in your Priors class ---

    def log_prior_A(self, Z_active: Set[FrozenSet[str]], A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int],
                    unit_drop: bool = True, num_admissible_edges: Optional[int] = None) -> float:
        """
        Computes log P(A | Z).
        Uses a fast O(1) calculation if num_admissible_edges is provided,
        otherwise falls back to an O(|Z|^2) calculation.
        """
        if num_admissible_edges is not None:
            # --- FAST PATH (for A-only MCMC) ---
            num_edges_present = len(A)
            log_rho = math.log(self.rho)
            log_one_minus_rho = math.log(1 - self.rho)
            
            logp = (num_edges_present * log_rho +
                    (num_admissible_edges - num_edges_present) * log_one_minus_rho)
            return logp
        else:
            # --- SLOW PATH (Fallback for Z-MCMC) ---
            logp = 0.0
            for P in Z_active:
                for Q in Z_active:
                    if admissible_edge(P, Q, unit_drop):
                        a = 1 if A.get((P, Q), 0) == 1 else 0
                        logp += math.log(self.rho) if a == 1 else math.log(1 - self.rho)
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

    def update_reach_add_edge(self, u: FrozenSet[str], v: FrozenSet[str]):
        """
        Incrementally and efficiently updates the Reach dictionary AFTER 
        an edge u -> v has been added to the graph A.
        """
        # 1. Find all nodes 'x' that can reach u (the "ancestors" of u).
        ancestors_of_u = {x for x, reach_set in self.Reach.items() if u in reach_set}
        
        # 2. Find all nodes 'y' that v can reach (the "descendants" of v).
        #    We use .get() for safety, though v should always be in Reach.
        descendants_of_v = self.Reach.get(v, set())
        
        # 3. For each ancestor 'x', add all of v's descendants to its reachability set.
        #    This creates all the new paths: (x -> ... -> u -> v -> ... -> y).
        for x in ancestors_of_u:
            self.Reach[x].update(descendants_of_v)

    def update_reach_remove_edge(self, u: FrozenSet[str], v: FrozenSet[str]):
        """
        Handles the Reach update for edge removal. For safety and simplicity,
        this falls back to a full re-computation.
        """
        self.recompute_reach()

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
    
# ==============================================================================
# === NEW VITERBI DP FUNCTIONS (for Phase 2) ===================================
# ==============================================================================

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
    F_full: Structure, # The fully-connected structure
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    leaf_type_maps: List[Dict[str, str]] # Needed for weighted flow
) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], float]:
    """
    Calculates the "Viterbi flow" w(P,Q) for all potential edges.
    """
    print("Calculating Viterbi flow w(P,Q) for all trees...")
    viterbi_flow = defaultdict(float)
    
    # Pre-calculate leaf counts for all nodes in all trees
    # (This is needed for the paper's flow weighting)
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

    for i, (tree, B_sets, leaf_counts) in enumerate(zip(trees, all_B_sets, leaf_counts_all_trees)):
        # 1. Run Viterbi DP for this tree
        root_table, memo = dp_tree_root_viterbi(
            tree, 
            F_full.labels_list, 
            F_full.Reach, 
            B_sets
        )
        
        # 2. Reconstruct the single best labeling
        L_MAP = find_best_viterbi_labeling(tree, root_table, memo)
        
        if not L_MAP:
            print(f"Warning: Could not find a valid Viterbi labeling for tree {i}.")
            continue
            
        # 3. Iterate over tree edges and aggregate flow
        for (v, u) in iter_edges(tree): # v=parent, u=child
            P = L_MAP.get(v)
            Q = L_MAP.get(u)
            
            if P is None or Q is None:
                continue # Node was not in the labeling (e.g., filtered leaf)
                
            if P != Q:
                # This is a transition. Add its flow.
                # Use the paper's weighting: number of leaf descendants of the *child*
                flow_weight = leaf_counts.get(u, 1) # Default to 1 if not found
                viterbi_flow[(P, Q)] += flow_weight
                
    return viterbi_flow
        
# ----------------------------
# Scoring: log posterior
# ----------------------------

def score_structure(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ACCEPT IT HERE
                    priors: Priors,
                    num_admissible_edges: Optional[int] = None,  # <--- ADD NEW ARGUMENT
                    precomputed_logp_Z: Optional[float] = None,  # <--- ADD NEW ARGUMENT
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # Compute the (log) posterior score of a candidate structure F = (Z_active, A).


    if precomputed_logp_Z is not None:
        logp_Z = precomputed_logp_Z
    else:
        logp_Z = priors.log_prior_Z(struct.S, struct.Z_active)
    
    if not math.isfinite(logp_Z):
        return float("-inf"), []
        
    logp_A = priors.log_prior_A(
        struct.Z_active, struct.A,
        unit_drop=struct.unit_drop,
        num_admissible_edges=num_admissible_edges
    )
    
    logp = logp_Z + logp_A
    # # ---- Prior over structure F ----
    # logp = priors.log_prior_Z(struct.S, struct.Z_active)
    # if not math.isfinite(logp):
    #     return float("-inf"), []
    # # Add the edge prior, which is the key difference in this function
    # logp += priors.log_prior_A(
    #         struct.Z_active, struct.A,
    #         unit_drop=struct.unit_drop,
    #         num_admissible_edges=num_admissible_edges # <--- PASS IT ALONG
    #     )
    # ---- Likelihood over all trees ----
    logLs = []
    for i, (root, leaf_to_type, B_sets) in enumerate(zip(trees, leaf_type_maps, all_B_sets)):
        # B_sets = compute_B_sets(root, leaf_to_type)
        # root_labels = B_sets.get(root, set())

        # if not root_labels:
        #     logLs.append(0.0)
        #     continue

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

# --- MODIFY the score_structure function ---
# (Keep the version relevant to your chosen MCMC: Z-only or Z+A)


def score_structure_no_edge_prior(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ACCEPT IT HERE
                    priors: Priors,
                    precomputed_logp_Z: float, # <--- CHANGE: This is now a required argument
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # print("\n" + "="*50)
    # print("=== STARTING SCORE_STRUCTURE ===")
    
    logp = precomputed_logp_Z # Use the provided value directly
    # print(f"[DEBUG score_structure] Log Prior P(Z): {logp}")
    if not math.isfinite(logp):
        # print("[DEBUG score_structure] P(Z) is -inf. ABORTING.")
        return float("-inf"), []

    logLs = []
    for i, (root, leaf_to_type, B_sets) in enumerate(zip(trees, leaf_type_maps, all_B_sets)):
        # print(f"\n--- Processing Tree {i+1}/{len(trees)} ---")
        # B_sets = compute_B_sets(root, leaf_to_type)
        # root_labels = B_sets.get(root, set())
        # # print(f"[DEBUG score_structure] Tree {i+1} has {len(root_labels)} unique types in its leaves.")

        # if not root_labels:
        #     # print(f"[DEBUG score_structure] Tree {i+1} has no mapped leaves. LogL = 0.0")
        #     logLs.append(0.0)
        #     continue
        
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

# REPLACEMENT for score_structure_no_edge_prior (used in Z-only MCMC)
def score_structure_no_edge_prior_collapsed(
    struct: Structure,
    collapsed_data: pd.DataFrame, # <-- Takes the loaded DataFrame
    priors: Priors,
    precomputed_logp_Z: float,
    prune_eps: float = 0.0
) -> Tuple[float, List[float]]:
    """ Calculates log P(Z) + sum [ count * log P(Tree | Z) ] using collapsed data."""

    logp_prior = precomputed_logp_Z
    if not math.isfinite(logp_prior):
        return float("-inf"), []

    total_log_likelihood = 0.0
    per_potency_weighted_logL = [] # Optional: Store weighted logL

    # Use the appropriate Reach object for Z-only MCMC
    reach_object = SubsetReach()

    for idx, row in collapsed_data.iterrows():
        cell_types_present = {col for col in collapsed_data.columns[:-1] if row[col] == 1}
        potency_set = frozenset(cell_types_present)
        count = row['counts']

        if count <= 0 or not potency_set: continue

        potency_id = str(idx)
        virtual_root, virtual_leaf_map = create_virtual_star_tree(potency_set, potency_id)
        B_sets_single = compute_B_sets(virtual_root, virtual_leaf_map)

        C_log = dp_tree_root_table(
            virtual_root,
            struct.labels_list,
            reach_object, # Use SubsetReach here
            B_sets_single,
            prune_eps=prune_eps
        )

        if not C_log: return float("-inf"), []

        tree_logL = tree_marginal_from_root_table_log(C_log)
        if not math.isfinite(tree_logL): return float("-inf"), []

        weighted_logL = tree_logL * count
        total_log_likelihood += weighted_logL
        per_potency_weighted_logL.append(weighted_logL)

    final_score = logp_prior + total_log_likelihood
    return final_score, per_potency_weighted_logL


# REPLACEMENT for score_structure (used in Z+A MCMC or final scoring)
def score_structure_collapsed(
    struct: Structure,
    collapsed_data: pd.DataFrame, # <-- Takes the loaded DataFrame
    priors: Priors,
    num_admissible_edges: Optional[int] = None,
    precomputed_logp_Z: Optional[float] = None,
    prune_eps: float = 0.0
) -> Tuple[float, List[float]]:
    """ Calculates log P(Z) + log P(A|Z) + sum [ count * log P(Tree | Z, A) ] """

    # Calculate log P(Z)
    if precomputed_logp_Z is not None:
        logp_Z = precomputed_logp_Z
    else:
        logp_Z = priors.log_prior_Z(struct.S, struct.Z_active)

    if not math.isfinite(logp_Z):
        return float("-inf"), []

    # Calculate log P(A|Z)
    print(f"Len of struct.A is = {len(struct.A)}")
    logp_A = priors.log_prior_A(
        struct.Z_active, struct.A,
        unit_drop=struct.unit_drop,
        num_admissible_edges=num_admissible_edges
    )
    if not math.isfinite(logp_A): # Should generally be finite unless rho is 0 or 1
        return float("-inf"), []

    logp = logp_Z + logp_A
    print(f"Log_p = {logp}")
    # total_log_likelihood = 0.0
    # per_potency_weighted_logL = [] # Optional
    logLs = []


    # Use the REAL Reach object calculated by the Structure class
    reach_object = struct.Reach

    for idx, row in collapsed_data.iterrows():
        cell_types_present = {col for col in collapsed_data.columns[:-1] if row[col] == 1}
        potency_set = frozenset(cell_types_present)
        count = row['counts']

        if count <= 0 or not potency_set: continue

        potency_id = str(idx)
        virtual_root, virtual_leaf_map = create_virtual_star_tree(potency_set, potency_id)
        B_sets_single = compute_B_sets(virtual_root, virtual_leaf_map)

        C_log = dp_tree_root_table(
            virtual_root,
            struct.labels_list,
            reach_object, # Use the real Reach object here
            B_sets_single,
            prune_eps=prune_eps
        )

        if not C_log: return float("-inf"), []

        tree_logL = tree_marginal_from_root_table_log(C_log)
        if not math.isfinite(tree_logL): return float("-inf"), []

        weighted_logL = (tree_logL + logp)* count
        # total_log_likelihood += weighted_logL
        # per_potency_weighted_logL.append(weighted_logL)
        logLs.append(weighted_logL)

    # final_score = logp_prior + total_log_likelihood
    total_log_post = sum(logLs)
    return total_log_post, logLs


# deterministically build ALL admissible edges for a given Z (keeps MCMC symmetric/easy)
def _full_edges_for_Z(Z_active: Set[FrozenSet[str]], unit_drop_edges: bool) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
    A = {}
    Zl = list(Z_active)
    for P in Zl:
        for Q in Zl:
            if admissible_edge(P, Q, unit_drop_edges):
                A[(P, Q)] = 1
    return A

# Place this function BEFORE mcmc_map_search in your script
def propose_swap_Z_and_update_A_incrementally(
    current_struct: "Structure",
    rng: random.Random,
    candidate_pool: List[FrozenSet[str]],
    block_swap_sizes: Tuple[int, ...],
    fitch_probs: Dict[FrozenSet[str], float]
) -> Tuple[Optional["Structure"], Optional[Dict]]:
    """
    Proposes swapping 'm' potencies and incrementally updates the edge set 'A'.
    This version is fixed to handle duplicate selections from random.choices.
    """
    root = frozenset(current_struct.S)
    acti = [P for P in current_struct.Z_active if len(P) >= 2 and P != root]
    ina = [P for P in candidate_pool if P not in current_struct.Z_active]

    if not acti or not ina:
        return None, None

    # m = rng.choice(block_swap_sizes)
    # m = max(1, min(m, len(acti), len(ina)))
    m = 1
    
    # --- Weighted sampling (this part is fine) ---
    add_weights = [fitch_probs.get(p, 0.001) for p in ina]
    drop_weights = [1.0 - fitch_probs.get(p, 0.999) for p in acti]
    
    add = random.choices(ina, weights=add_weights, k=m)
    drop = random.choices(acti, weights=drop_weights, k=m)

    new_struct = current_struct.clone()

    #
    # V V V THIS IS THE FIX V V V
    #
    # By converting `drop` to a `set`, we automatically remove any duplicates
    # before we start iterating, preventing the KeyError.
    for P_rm in set(drop):
        new_struct.Z_active.remove(P_rm)
    
    # Similarly, ensure `add` potencies are unique to avoid issues
    for P_add in set(add):
        new_struct.Z_active.add(P_add)
    
    # Prune edges related to ANY of the dropped potencies
    drop_set = set(drop)
    new_struct.A = {edge: 1 for edge in new_struct.A if not (edge[0] in drop_set or edge[1] in drop_set)}
    
    # Wire up the newly added (and unique) potencies
    for P_add in set(add):
        for P_other in new_struct.Z_active:
            if P_add == P_other: continue
            if admissible_edge(P_add, P_other, new_struct.unit_drop):
                new_struct.A[(P_add, P_other)] = 1
            if admissible_edge(P_other, P_add, new_struct.unit_drop):
                new_struct.A[(P_other, P_add)] = 1

    # new_struct.recompute_reach()
    
    # The details can still show the original samples if you wish
    details = {'drop': drop, 'add': add}
    return new_struct, details


# ==============================================================================
# === ADD THIS NEW HELPER FUNCTION =============================================
# ==============================================================================

def _print_full_state(struct: Structure, score: float, title: str):
    """A helper function to pretty-print the current MCMC state."""
    print(f"\n{title}")
    print(f"Score: {score:.4f}")
    
    # Get all potencies, separating singletons
    S = {frozenset([t]) for t in struct.S}
    multi_sorted = sorted(
        [P for P in struct.Z_active if P not in S],
        key=lambda x: (len(x), tuple(sorted(list(x))))
    )
    print(f"Active Potencies (k={len(multi_sorted)}):")
    for P in multi_sorted: 
        print(f"  {pot_str(P)}")

    print(f"Active Edges ({len(struct.A)}):")
    edges = sorted(
        [e for e, v in struct.A.items() if v == 1],
        key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1]))
    )
    for P, Q in edges: 
        print(f"  {pot_str(P)} -> {pot_str(Q)}")
    print("-" * 60)

# ====== MCMC over Z (potency sets). A is deterministic (all admissible edges). ======
# Initial mcmc map search (no fitch, pure random)
def mcmc_map_search(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str,str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ADD HERE
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
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
    Metropolis–Hastings sampler that explores Z; A is rebuilt deterministically as all admissible edges.
    Target:  log P(Z) + sum_T log P(T | Z)    (edge prior removed)
    Proposals are symmetric -> accept with min(1, exp(delta)).
    """

    def make_struct_no_reach(Zset: Set[FrozenSet[str]]) -> "Structure":
        A = _full_edges_for_Z(Zset, unit_drop_edges)
        struct = Structure(S, Zset, A, unit_drop=unit_drop_edges)
        # OVERWRITE the expensive transitive closure with our dummy object
        struct.Reach = SubsetReach()
        return struct

    rng = random.Random(seed)

    # ----- This section is unchanged -----
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
        if not candidate_pool:
            candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    pool_set = set(candidate_pool)

    # ----- Initialization logic is unchanged -----
    singles = {frozenset([t]) for t in S}
    root = frozenset(S)
    if priors.potency_mode == "fixed_k" and fixed_k is not None:
        try:
            _, Z0 = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
            multis0 = [P for P in Z0 if len(P) >= 2]
            if len(multis0) != fixed_k:
                raise RuntimeError("Fitch init gave wrong k")
        except Exception:
            Z0 = set(singles); Z0.add(root)
            available = [P for P in candidate_pool if P != root]
            if len(available) < max(0, fixed_k - 1):
                raise RuntimeError("Not enough candidates to seed fixed-k")
            Z0.update(rng.sample(available, fixed_k - 1))
    else:
        Z0 = set(singles); Z0.add(root)
        sprinkle = min( max(0, len(candidate_pool)//20), 5 )
        if sprinkle:
            Z0.update(rng.sample(candidate_pool, sprinkle))
    
    current = make_struct_no_reach(Z0)
    current_logp_Z = priors.log_prior_Z(S, current.Z_active)
    curr_score, _ = score_structure_no_edge_prior(current, trees, leaf_type_maps,all_B_sets, priors, precomputed_logp_Z=current_logp_Z)
    if not math.isfinite(curr_score):
        # for _ in range(20):
        #     # Fallback logic is unchanged
        #     if priors.potency_mode == "fixed_k" and fixed_k is not None:
        #         Z0 = set(singles); Z0.add(root)
        #         Z0.update(rng.sample([P for P in candidate_pool if P != root], fixed_k - 1))
        #     else:
        #         Z0 = set(singles); Z0.add(root)
        #         Z0.update(rng.sample(candidate_pool, min(5, len(candidate_pool))))
        #     current = make_struct(Z0)
        #     curr_score, _ = score_structure_no_edge_prior(current, trees, leaf_type_maps, priors)
        #     if math.isfinite(curr_score):
        #         break
        if not math.isfinite(curr_score):
            raise RuntimeError("Could not find a finite-scoring starting point for MCMC.")


    # ----- proposal kernels (all symmetric) -----
    def propose_fixed_k_swap(Zset: Set[FrozenSet[str]]) -> Tuple[Optional[Set[FrozenSet[str]]], Optional[Dict]]:
        root = frozenset(S)
        acti = [P for P in Zset if len(P) >= 2]
        act = [P for P in acti if P!= root]
        ina = [P for P in candidate_pool if P not in Zset]
        if not act or not ina:
            return None, None

        # m = rng.choice(block_swap_sizes)
        # m = max(1, min(m, len(act), len(ina)))
        m = 1

        # Weighted sampling for ADDING potencies
        add_weights = [fitch_probs.get(p, 0.001) for p in ina]
        add = random.choices(ina, weights=add_weights, k=m) if sum(add_weights) > 0 else rng.sample(ina, m)

        # Weighted sampling for DROPPING potencies
        drop_weights = [1.0 - fitch_probs.get(p, 0.999) for p in act]
        drop = random.choices(act, weights=drop_weights, k=m) if sum(drop_weights) > 0 else rng.sample(act, m)

        Z2 = set(Zset)

        # <<<--- THIS IS THE FIX --->>>
        # Use set operations which are robust to duplicates in the 'drop' and 'add' lists.
        Z2.difference_update(drop)
        Z2.update(add)
    
        details = {'drop': drop, 'add': add}
        return Z2, details
    
    # This function will go INSIDE mcmc_map_search
    def propose_jaccard_swap(Zset: Set[FrozenSet[str]]) -> Tuple[Optional[Set[FrozenSet[str]]], Optional[Dict]]:
        """
        Proposes swapping one active potency (excluding the root) with an inactive one.
        The inactive potency is chosen with a probability weighted by the inverse of
        the Jaccard distance to the dropped potency.
        """
        act = [P for P in Zset if len(P) >= 2]
        ina = [P for P in candidate_pool if P not in Zset]
        root = frozenset(S)

        # i) Create a list of candidates to drop, excluding the root potency
        drop_candidates = [P for P in act if P != root]

        if not drop_candidates or not ina:
            return None, None

        # Pick one potency to drop uniformly at random from the candidates
        P_drop = rng.choice(drop_candidates)

        # ii) Calculate weights for the inactive candidates
        # The weight is the inverse of the Jaccard distance.
        # A small epsilon prevents division by zero if the distance is 0.
        epsilon = 1e-6
        jaccard_weights = [1.0 / (jaccard_distance(P_drop, p_add) + epsilon) for p_add in ina]

        # Choose the potency to add based on the calculated weights
        # random.choices returns a list, so we take the first element
        P_add = random.choices(ina, weights=jaccard_weights, k=1)[0]
        
        # Perform the swap
        Z2 = set(Zset)
        Z2.remove(P_drop)
        Z2.add(P_add)

        # Prepare details for logging
        details = {'drop': [P_drop], 'add': [P_add]}
        return Z2, details
    

    def propose_toggle(Zset: Set[FrozenSet[str]]) -> Optional[Set[FrozenSet[str]]]:
        # Toggling a potency "on" should favor high-probability candidates
        # Toggling "off" should favor low-probability candidates
        # For simplicity, we'll just use the weights to pick a candidate to consider toggling.
        all_candidates = list(pool_set)
        weights = [fitch_probs.get(p, 0.001) for p in all_candidates]
        
        P = random.choices(all_candidates, weights=weights, k=1)[0]
        
        Z2 = set(Zset)
        if P in Z2:
            Z2.remove(P)
        else:
            Z2.add(P)
        return Z2

    # ----- MCMC loop -----
    kept_Z = []
    kept_scores = []
    best_struct = current.clone()
    best_score = curr_score


    all_scores_trace = [] # <--- **ADDED**: List to store score at EVERY iteration

    accepts = 0
    tried = 0
    
    # Helper for pretty printing
    def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (Z-only)", leave=True)

    all_scores_trace.append(curr_score)

    for it in iterator:
        Zprop = None
        # <<<--- CHANGE 2: Add a variable to store proposal details --->>>
        proposal_details = None

        if priors.potency_mode == "fixed_k" and fixed_k is not None:
            # <<<--- CHANGE 3: Unpack the details from the proposal call --->>>
            # Zprop, proposal_details = propose_fixed_k_swap(current.Z_active)
            # Zprop, proposal_details = propose_jaccard_swap(current.Z_active)

            # if proposal_details:
            #     dropped_str = ", ".join(pot_str(p) for p in proposal_details['drop'])
            #     added_str = ", ".join(pot_str(p) for p in proposal_details['add'])
            #     print(f"\n[Proposing Swap]: {dropped_str} -> {added_str}")
            prop_struct, proposal_details = propose_swap_Z_and_update_A_incrementally(
                current_struct=current,
                rng=rng,
                candidate_pool=candidate_pool,
                block_swap_sizes=block_swap_sizes,
                fitch_probs=fitch_probs
            )
            proposed_logp_Z = current_logp_Z 

            if prop_struct:
                # Overwrite the Reach object on the new proposal
                prop_struct.Reach = SubsetReach()
        else:
            Zprop = propose_toggle(current.Z_active)

        # if Zprop is None:
        #     tried += 1
        #     if progress:
        #         iterator.set_postfix({"logpost": f"{curr_score:.3f}", "acc": f"{(accepts/max(1,tried)):.2f}"})
        #     continue

        if prop_struct is None:
            tried += 1
            if progress:
                iterator.set_postfix({
                    "logpost": f"{curr_score:.3f}",
                    "acc": f"{(accepts/max(1,tried)):.2f}"
                })
            continue

        # prop_struct = make_struct(Zprop)
        # prop_score, _ = score_structure_no_edge_prior(prop_struct, trees, leaf_type_maps, all_B_sets, priors)
        
        prop_score, _ = score_structure_no_edge_prior(prop_struct, trees, leaf_type_maps, all_B_sets, priors, proposed_logp_Z)

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta))

        tried += 1
        if accept:
            accepts += 1
            
            # <<<--- CHANGE 4: Add the logging logic for accepted swaps --->>>
            if proposal_details:
                dropped_str = ", ".join(pot_str(p) for p in proposal_details['drop'])
                added_str = ", ".join(pot_str(p) for p in proposal_details['add'])
                # Using print() will show the message on a new line below the progress bar
                # print(f"\n[Accepted Swap]: {dropped_str} -> {added_str} | New Score: {prop_score:.3f}")

            current = prop_struct
            curr_score = prop_score
            current_logp_Z = proposed_logp_Z # IMPORTANT: Update the prior score
            
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score

        all_scores_trace.append(curr_score)

        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_Z.append({P for P in current.Z_active if len(P) >= 2})
            kept_scores.append(curr_score)

        if progress:
            iterator.set_postfix({
                "logpost": f"{curr_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc": f"{(accepts/max(1,tried)):.2f}"
            })
    
    # ----- Final stats calculation (unchanged) -----
    counts: Dict[FrozenSet[str], int] = {P: 0 for P in pool_set}
    for Zs in kept_Z:
        for P in Zs:
            if P in counts:
                counts[P] += 1
    total_kept = max(1, len(kept_Z))
    inclusion = {P: counts[P] / total_kept for P in counts}

    stats = {
        "samples": kept_Z,
        "scores": kept_scores,
        "all_scores_trace": all_scores_trace, # <--- **ADDED**: The full trace
        "accept_rate": (accepts / max(1, tried)),
        "inclusion": inclusion,
    }
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
    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ADD HERE
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

    all_admissible_pairs = [
        (P, Q)
        for P in Z
        for Q in Z
        if admissible_edge(P, Q, unit_drop_edges)
    ]

    num_admissible_edges = len(all_admissible_pairs) # Get the total count once

    logp_Z_fixed = priors.log_prior_Z(S, Z)

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
        return build_mid_sized_connected_dag(Z, keep_prob=0.3, rng=rng)

    # Z = seed_Z()
    A = seed_A(Z)
    current = Structure(S, Z, A, unit_drop=unit_drop_edges)
    curr_score, _ = score_structure(current, trees, leaf_type_maps, all_B_sets, priors, num_admissible_edges=num_admissible_edges,precomputed_logp_Z=logp_Z_fixed)
    if not math.isfinite(curr_score):
        # rescue a few times
        for _ in range(20):
            # Z = seed_Z(); 
            A = seed_A(Z)
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, num_admissible_edges=num_admissible_edges,precomputed_logp_Z=logp_Z_fixed)
            if math.isfinite(curr_score):
                break
        if not math.isfinite(curr_score):
            raise RuntimeError("Could not find a finite-scoring starting point for MCMC.")

    best_struct = current.clone()
    best_score = curr_score


    if not all_admissible_pairs:
        print("[Warning] No admissible edges found for the given Z_fixed set. MCMC will not move.")


    # ---------- symmetric proposal kernels ----------
    def admissible_pairs(Zset: Set[FrozenSet[str]]) -> List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        L = list(Zset)
        out = []
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, unit_drop_edges):
                    out.append((P, Q))
        return out


    # ---------- MCMC main loop ----------
    kept_Z = []
    kept_scores = []
    kept_edge_density = []

    all_scores_trace = [] # <--- **ADDED**: List to store score at EVERY iteration

    accepts = 0
    tried = 0

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (A-only)", leave=True)

    all_scores_trace.append(curr_score)

    for it in iterator:


        if not all_admissible_pairs:
            # If there are no possible edges, we can't do anything.
            break

        # 1. Propose a change by picking a random edge to flip
        edge_to_flip = rng.choice(all_admissible_pairs)
        u, v = edge_to_flip
        
        action = 'added' if current.A.get(edge_to_flip, 0) == 0 else 'removed'
        last_proposal_details = {'edge': edge_to_flip, 'action': action}
        
        prop = current.clone()
        
        # Apply the toggle based on the action
        if action == 'added':
            # Use the fast, incremental update for additions.
            prop.A[edge_to_flip] = 1  # <-- YOU WERE MISSING THIS
            prop.update_reach_add_edge(u, v)
        else: # action == 'removed'
            # Use the safe, full re-computation for removals.
            del prop.A[edge_to_flip]  # <-- YOU WERE MISSING THIS
            prop.update_reach_remove_edge(u, v)
        
        # prop.recompute_reach()

        # 2. Score the new structure using the parallel scorer
        prop_score, _ = score_structure(prop, trees, leaf_type_maps, all_B_sets, priors, num_admissible_edges=num_admissible_edges, precomputed_logp_Z=logp_Z_fixed)
        
        # prop = None
        # last_proposal_details = None
        
        # prop, last_proposal_details = prop_edge_toggle(current, rng)
        # # prop = prop_edge_toggle(current)

        # if prop is None:
        #     # count as tried but no change
        #     tried += 1
        #     if progress:
        #         iterator.set_postfix({
        #             "logpost": f"{curr_score:.3f}",
        #             "best": f"{best_score:.3f}",
        #             "acc": f"{(accepts/max(1,tried)):.2f}",
        #             "E": f"{sum(prop.A.values()) if prop else sum(current.A.values())}"
        #         })
        #     continue

        # prop_score, _ = score_structure(prop, trees, leaf_type_maps, all_B_sets, priors)

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta))

        tried += 1
        if accept:
            accepts += 1

            # if last_proposal_details and 'edge' in last_proposal_details:
            #     edge_details = last_proposal_details
            #     edge_str = f"{pot_str(edge_details['edge'][0])} -> {pot_str(edge_details['edge'][1])}"
            #     action_str = edge_details['action'].capitalize()
            #     print(f"[Accepted Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")

            current = prop
            curr_score = prop_score
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score

        all_scores_trace.append(curr_score)

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
        "all_scores_trace": all_scores_trace, # <--- **ADDED**: The full trace
        "edge_density": kept_edge_density,
        "accept_rate": accepts / max(1, tried),
        "inclusion": inclusion,
    }
    # print(f"Accepts = {accepts}")
    return best_struct, best_score, stats


# ==============================================================================
# === 1. NEW LIKELIHOOD-ONLY SCORING FUNCTION ==================================
# ==============================================================================

def get_log_likelihood(
    struct: Structure,
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    prune_eps: float = 0.0
) -> float:
    """
    Calculates *only* the total log-likelihood term: Sum_i log P(T_i | F).
    It does NOT include any priors P(Z) or P(A|Z).
    """
    total_log_L = 0.0
    
    # We use the struct's pre-computed Reach and labels_list
    active_labels = struct.labels_list
    Reach = struct.Reach
    
    for i, (root, leaf_to_type, B_sets) in enumerate(zip(trees, leaf_type_maps, all_B_sets)):
        
        # 1. Call the DP function that returns a log-space table.
        C_log = dp_tree_root_table(
            root, 
            active_labels, 
            Reach, 
            B_sets, 
            prune_eps=prune_eps
        )

        if not C_log:
            # This structure is impossible for this tree.
            return -math.inf

        # 2. Call the marginalization function that works entirely in log-space.
        tree_logL = tree_marginal_from_root_table_log(C_log)

        if not math.isfinite(tree_logL):
            # If the final log-likelihood is not a valid number, abort.
            return -math.inf

        # 3. Accumulate the total log-likelihood
        total_log_L += tree_logL
        
    return total_log_L

# ==============================================================================
# === 2. HELPERS FOR "CLEVER" PROPOSALS ========================================
# ==============================================================================

def _propose_clever_A(
    current_struct: Structure, 
    fitch_probs: Dict[FrozenSet[str], float],
    rng: random.Random
) -> Tuple[Optional[Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]], float, float]:
    """
    Proposes flipping ONE edge, biased by Fitch probabilities.
    This is an ASYMMETRIC proposal and requires a Hastings correction.
    
    Returns:
        (A_prop, log_q_forward, log_q_reverse)
    """
    all_admissible_edges = current_struct.all_edge_pairs()
    if not all_admissible_edges:
        return None, 0.0, 0.0, None, None

    current_A = current_struct.A
    
    # --- 1. Calculate weights for ALL admissible edges ---
    weights_fwd = []
    log_weights_fwd = []
    
    # Get a baseline probability for edges not in fitch_probs (e.g., singletons)
    # Use a small non-zero value
    min_prob = 1e-6 

    for edge in all_admissible_edges:
        P, Q = edge
        # Get Fitch prob for the *parent* node, default to min_prob
        p_fitch = fitch_probs.get(P, min_prob)
        
        if current_A.get(edge, 0) == 1:
            # Edge is ACTIVE. Proposal weight is to DROP it.
            # We want to drop edges with LOW support.
            weight = max(min_prob, 1.0 - p_fitch)
        else:
            # Edge is INACTIVE. Proposal weight is to ADD it.
            # We want to add edges with HIGH support.
            weight = max(min_prob, p_fitch)
        
        weights_fwd.append(weight)
        log_weights_fwd.append(math.log(weight))

    Z_fwd = sum(weights_fwd)
    if Z_fwd == 0: # Should not happen with min_prob
        return None, 0.0, 0.0, None, None
        
    log_Z_fwd = math.log(Z_fwd)

    # --- 2. Propose the edge to flip ---
    chosen_idx = rng.choices(range(len(all_admissible_edges)), weights=weights_fwd, k=1)[0]
    edge_to_flip = all_admissible_edges[chosen_idx]
    
    A_prop = dict(current_A)
    action: str
    if current_A.get(edge_to_flip, 0) == 1:
        del A_prop[edge_to_flip]
        action = 'removed'
    else:
        A_prop[edge_to_flip] = 1
        action = 'added'

    # --- 3. Calculate log_q_forward ---
    log_q_fwd = log_weights_fwd[chosen_idx] - log_Z_fwd

    # --- 4. Calculate log_q_reverse ---
    # We are now in state A_prop. What's the prob of flipping `edge_to_flip` back?
    
    # Get the *reverse* weight for the chosen edge
    P_flip, Q_flip = edge_to_flip
    p_fitch_flip = fitch_probs.get(P_flip, min_prob)
    
    weight_rev_edge: float
    if action == 'added':
        # Reverse move is to REMOVE it. Weight is (1 - p_fitch)
        weight_rev_edge = max(min_prob, 1.0 - p_fitch_flip)
    else:
        # Reverse move is to ADD it. Weight is p_fitch
        weight_rev_edge = max(min_prob, p_fitch_flip)

    # The *new* normalization constant Z_rev
    Z_rev = Z_fwd - weights_fwd[chosen_idx] + weight_rev_edge
    log_Z_rev = math.log(Z_rev)
    
    log_q_rev = math.log(weight_rev_edge) - log_Z_rev

    return A_prop, log_q_fwd, log_q_rev, edge_to_flip, action


def _propose_clever_Z(
    current_struct: Structure,
    rho: float, # priors.rho
    rng: random.Random,
    candidate_pool: List[FrozenSet[str]]
) -> Tuple[Optional[Set[FrozenSet[str]]], Optional[Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]], float, float]:
    """
    Proposes a (Z', A') jointly.
    1. Symmetrically swaps one potency Z -> Z'.
    2. Stochastically re-wires edges for the new potency to get A'.
    
    This is an ASYMMETRIC proposal.
    
    Returns:
        (Z_prop, A_prop, log_q_forward_rewire, log_q_reverse_rewire, P_d, P_a)
    """
    S = current_struct.S
    Z = current_struct.Z_active
    A = current_struct.A
    unit_drop = current_struct.unit_drop
    root = frozenset(S)
    
    # --- 1. Symmetric Z-Swap Proposal ---
    acti = [P for P in Z if len(P) >= 2 and P != root]
    ina = [P for P in candidate_pool if P not in Z] # Assumes pool has no root/singletons
    
    if not acti or not ina:
        return None, None, 0.0, 0.0, None, None
        
    P_d = rng.choice(acti) # Potency to drop
    P_a = rng.choice(ina)  # Potency to add
    
    Z_prop = (Z - {P_d}) | {P_a}
    
    # Since we picked P_d and P_a uniformly, the Z-swap part is symmetric.
    # Q_swap(Z'|Z) = 1/|acti| * 1/|ina|
    # Q_swap(Z|Z') = 1/|acti'| * 1/|ina'| 
    # Here |acti'| = |acti| and |ina'| = |ina|. So Q_swap is symmetric.
    # The Hastings ratio only depends on the *rewiring* step.
    
    log_rho = math.log(rho)
    log_one_minus_rho = math.log(1 - rho)

    # --- 2. Stochastic Rewire (Forward) ---
    # Propose A' based on Z_prop and P_a
    
    A_pruned = {e: v for e, v in A.items() if P_d not in e}
    A_prop = dict(A_pruned)
    log_q_fwd_rewire = 0.0

    # Find all potential edges for the *new* node P_a within the *new* set Z_prop
    potential_edges_fwd = []
    for P_other in Z_prop:
        if P_other == P_a: continue
        # R -> P_a
        if admissible_edge(P_other, P_a, unit_drop):
            potential_edges_fwd.append((P_other, P_a))
        # P_a -> Q
        if admissible_edge(P_a, P_other, unit_drop):
            potential_edges_fwd.append((P_a, P_other))
            
    # For each potential edge, flip a coin
    for edge in potential_edges_fwd:
        if rng.random() < rho:
            A_prop[edge] = 1
            log_q_fwd_rewire += log_rho
        else:
            # No edge, do nothing to A_prop
            log_q_fwd_rewire += log_one_minus_rho

    # --- 3. Calculate Reverse Probability ---
    # What's the probability of stochastically generating the *original*
    # edges connected to P_d?
    
    log_q_rev_rewire = 0.0
    
    # Find all potential edges for the *old* node P_d within the *old* set Z
    potential_edges_rev = []
    for P_other in Z:
        if P_other == P_d: continue
        # R -> P_d
        if admissible_edge(P_other, P_d, unit_drop):
            potential_edges_rev.append((P_other, P_d))
        # P_d -> Q
        if admissible_edge(P_d, P_other, unit_drop):
            potential_edges_rev.append((P_d, P_other))

    # Calculate the probability of generating the *exact* set of edges
    # that were connected to P_d in the original graph A.
    for edge in potential_edges_rev:
        if A.get(edge, 0) == 1:
            # This edge *existed*. Prob of generating it is rho.
            log_q_rev_rewire += log_rho
        else:
            # This edge *did not exist*. Prob of *not* generating it is (1-rho).
            log_q_rev_rewire += log_one_minus_rho

    return Z_prop, A_prop, log_q_fwd_rewire, log_q_rev_rewire, P_d, P_a

# ==============================================================================
# === 3. THE NEW MCMC FUNCTION (Metropolis-within-Gibbs) =======================
# ==============================================================================

def mcmc_map_search_clever_mwg(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    steps: int = 6000,
    burn_in: int = 1500,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    fitch_probs: Optional[Dict[FrozenSet[str], float]] = None,
    # New params to control inner loops
    a_step_inner_loops: int = 5,
    z_step_inner_loops: int = 1
) -> Tuple["Structure", float, Dict]:
    """
    Samples over Z and A using Metropolis-within-Gibbs with "clever"
    asymmetric proposals and the correct Metropolis-Hastings acceptance.
    
    Target Distribution: log P(Z) + log P(A|Z) + log L(Data|Z,A)
    """
    rng = random.Random(seed)

    # --- 1. Initialization ---
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
    
    if fitch_probs is None:
        fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
        fitch_probs = {p: prob for p, prob in fitch_probs_list}
        if not fitch_probs: fitch_probs = {} # Handle empty case

    # Seed state (re-using your logic)
    singles = {frozenset([t]) for t in S}
    root = frozenset(S)
    if priors.potency_mode == "fixed_k" and fixed_k is not None:
        Z0 = set(singles); Z0.add(root)
        available = [P for P in candidate_pool if P != root]
        needed = max(0, fixed_k - 1)
        if len(available) < needed:
             raise RuntimeError(f"Not enough candidates ({len(available)}) to seed fixed_k={fixed_k}")
        Z0.update(rng.sample(available, needed))
    else:
        # Fallback for Bernoulli
        Z0 = set(singles); Z0.add(root)
        available = [P for P in candidate_pool if P != root]
        sprinkle = min(5, len(available))
        if sprinkle:
            Z0.update(rng.sample(available, sprinkle))

    A0 = build_mid_sized_connected_dag(Z0, keep_prob=0.35, rng=rng)
    current_struct = Structure(S, Z0, A0, unit_drop=unit_drop_edges)

    # --- 2. Initial State Scoring ---
    logp_Z_curr = priors.log_prior_Z(S, current_struct.Z_active)
    
    all_admiss_curr = current_struct.all_edge_pairs()
    num_admiss_curr = len(all_admiss_curr)
    admiss_set_curr = set(all_admiss_curr) # For quick lookup
    
    logp_A_curr = priors.log_prior_A(
        current_struct.Z_active, current_struct.A, 
        unit_drop=unit_drop_edges,
        num_admissible_edges=num_admiss_curr
    )
    
    log_L_curr = get_log_likelihood(
        current_struct, trees, leaf_type_maps, all_B_sets
    )
    
    current_score = logp_Z_curr + logp_A_curr + log_L_curr
    
    if not math.isfinite(current_score):
        # Add retry logic if needed, omitted here for brevity
        raise RuntimeError("Could not find a finite-scoring starting point for MCMC.")
    
    # === ADDED: PRINT INITIAL STATE ===
    _print_full_state(current_struct, current_score, "=== INITIAL MCMC STATE ===")

    best_struct = current_struct.clone()
    best_score = current_score

    # --- 3. MCMC Loop Stats ---
    kept_Z = []
    kept_scores = []
    all_scores_trace = [current_score]
    accepts_A = 0
    tried_A = 0
    accepts_Z = 0
    tried_Z = 0

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (Clever MwG)", leave=True)

    # --- 4. Main MCMC Loop ---
    for it in iterator:
        
        # === Step 1: A-Step (Metropolis-within-Gibbs) ===
        # Propose a new A, given Z is fixed at current_struct.Z_active
        
        for _ in range(a_step_inner_loops):
            tried_A += 1

            A_prop, log_q_fwd, log_q_rev, edge_flipped, action_A = _propose_clever_A(
                current_struct, fitch_probs, rng
            )
            
            if A_prop is None:
                continue

            # Create the proposed structure (Z is unchanged)
            prop_struct_A = Structure(S, current_struct.Z_active, A_prop, unit_drop_edges)
            
            # Score the proposal
            logp_Z_prop = logp_Z_curr # Z is fixed
            logp_A_prop = priors.log_prior_A(
                prop_struct_A.Z_active, prop_struct_A.A,
                unit_drop=unit_drop_edges,
                num_admissible_edges=num_admiss_curr # Num admissible is fixed
            )
            log_L_prop = get_log_likelihood(
                prop_struct_A, trees, leaf_type_maps, all_B_sets
            )
            
            prop_score = logp_Z_prop + logp_A_prop + log_L_prop
            
            if math.isfinite(prop_score):
                delta_score = prop_score - current_score
                log_hastings = log_q_rev - log_q_fwd
                log_alpha = min(0.0, delta_score + log_hastings)
                
                if (log_alpha >= 0.0) or (rng.random() < math.exp(log_alpha)):
                    # === ADDED: PRINT A-STEP ACCEPTANCE ===
                    print(f"\n[ACCEPT A-Step | Iter {it}] Score: {current_score:.4f} -> {prop_score:.4f} (delta={delta_score:.4f})")
                    print(f"  Action: {action_A.capitalize()} edge {pot_str(edge_flipped[0])} -> {pot_str(edge_flipped[1])}")
                    accepts_A += 1
                    current_struct = prop_struct_A
                    current_score = prop_score
                    logp_A_curr = logp_A_prop # Z and num_admiss are same
                    log_L_curr = log_L_prop
                    # logp_Z_curr is unchanged


        # === Step 2: Z-Step (Metropolis-within-Gibbs) ===
        # Propose a new (Z, A) jointly
        
        for _ in range(z_step_inner_loops):
            tried_Z += 1

            Z_prop, A_prop, log_q_fwd_rewire, log_q_rev_rewire, P_dropped, P_added = _propose_clever_Z(
                current_struct, priors.rho, rng, candidate_pool
            )
            
            if Z_prop is None:
                continue
            
            # Create the proposed structure
            prop_struct_Z = Structure(S, Z_prop, A_prop, unit_drop_edges)

            # Score the proposal
            logp_Z_prop = priors.log_prior_Z(S, prop_struct_Z.Z_active)
            if not math.isfinite(logp_Z_prop):
                continue # Invalid Z (e.g., wrong k)
            
            all_admiss_prop = prop_struct_Z.all_edge_pairs()
            num_admiss_prop = len(all_admiss_prop)
            
            logp_A_prop = priors.log_prior_A(
                prop_struct_Z.Z_active, prop_struct_Z.A,
                unit_drop=unit_drop_edges,
                num_admissible_edges=num_admiss_prop
            )
            
            log_L_prop = get_log_likelihood(
                prop_struct_Z, trees, leaf_type_maps, all_B_sets
            )
            
            prop_score = logp_Z_prop + logp_A_prop + log_L_prop

            if math.isfinite(prop_score):
                delta_score = prop_score - current_score
                log_hastings = log_q_rev_rewire - log_q_fwd_rewire
                log_alpha = min(0.0, delta_score + log_hastings)

                if (log_alpha >= 0.0) or (rng.random() < math.exp(log_alpha)):

                    # === ADDED: PRINT Z-STEP ACCEPTANCE ===
                    print(f"\n[ACCEPT Z-Step | Iter {it}] Score: {current_score:.4f} -> {prop_score:.4f} (delta={delta_score:.4f})")
                    print(f"  Action: Dropped {pot_str(P_dropped)}")
                    print(f"  Action: Added   {pot_str(P_added)}")

                    accepts_Z += 1
                    current_struct = prop_struct_Z
                    current_score = prop_score
                    # Update all components of the score
                    logp_Z_curr = logp_Z_prop
                    logp_A_curr = logp_A_prop
                    log_L_curr = log_L_prop
                    num_admiss_curr = num_admiss_prop
                    admiss_set_curr = set(all_admiss_prop)


        # === Step 3: Post-Iteration Updates ===
        all_scores_trace.append(current_score)
        
        if current_score > best_score:
            best_struct = current_struct.clone()
            best_score = current_score
            
        # Collect sample
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_Z.append({P for P in current_struct.Z_active if len(P) >= 2})
            kept_scores.append(current_score)

        if progress:
            iterator.set_postfix({
                "logpost": f"{current_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc_A": f"{(accepts_A/max(1,tried_A)):.2f}",
                "acc_Z": f"{(accepts_Z/max(1,tried_Z)):.2f}",
                "|Z|": len(current_struct.Z_active) - len(S),
                "|A|": len(current_struct.A)
            })

    # === ADDED: PRINT FINAL STATE ===
    _print_full_state(best_struct, best_score, "=== FINAL BEST MCMC STATE ===")


    # --- 5. Final Stats Calculation ---
    counts = {P: 0 for P in candidate_pool}
    for Zs in kept_Z:
        for P in Zs:
            if P in counts:
                counts[P] += 1
    total_kept = max(1, len(kept_Z))
    inclusion = {P: counts[P] / total_kept for P in counts}

    stats = {
        "samples_Z": kept_Z,
        "scores": kept_scores,
        "all_scores_trace": all_scores_trace,
        "accept_rate_A": accepts_A / max(1, tried_A),
        "accept_rate_Z": accepts_Z / max(1, tried_Z),
        "accept_rate_overall": (accepts_A + accepts_Z) / max(1, tried_A + tried_Z),
        "inclusion": inclusion,
    }
    return best_struct, best_score, stats

def _mcmc_worker_clever_mwg(args: tuple):
    """
    A simple worker function that unpacks arguments and calls the
    new Metropolis-within-Gibbs MCMC sampler.
    This is the target for each parallel process.
    """
    # Unpack all the arguments. Order MUST match the 'tasks' tuple below.
    (
        S, trees, leaf_type_maps, all_B_sets, priors,
        unit_drop_edges, fixed_k, steps, burn_in, thin, 
        seed, progress,  # 'progress' will be passed as False for workers
        candidate_pool, fitch_probs,
        a_step_inner_loops, z_step_inner_loops
    ) = args

    # Call the new MCMC sampler
    return mcmc_map_search_clever_mwg(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        all_B_sets=all_B_sets,
        priors=priors,
        unit_drop_edges=unit_drop_edges,
        fixed_k=fixed_k,
        steps=steps,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        progress=progress, # This should be False to avoid console spam
        candidate_pool=candidate_pool,
        fitch_probs=fitch_probs,
        a_step_inner_loops=a_step_inner_loops,
        z_step_inner_loops=z_step_inner_loops
    )

# ==============================================================================
# === 2. PARALLEL MCMC RUNNER ==================================================
# ==============================================================================

def run_mcmc_clever_mwg_parallel(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]],
    priors: "Priors",
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    steps: int = 6000,
    burn_in: int = 1500,
    thin: int = 10,
    n_chains: int = 4,        # Replaces 'restarts'
    base_seed: int = 123,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    fitch_probs: Optional[Dict[FrozenSet[str], float]] = None,
    a_step_inner_loops: int = 5,
    z_step_inner_loops: int = 1
):
    """
    Runs multiple "Clever MwG" MCMC chains in parallel
    and returns the best result found.
    """
    if n_chains <= 0:
        n_chains = max(1, os.cpu_count() - 1)

    # Generate unique seeds for each independent chain
    seeds = [base_seed + i for i in range(n_chains)]

    # Package arguments for each worker. Each worker gets a different seed.
    tasks = []
    for seed in seeds:
        tasks.append((
            S, trees, leaf_type_maps, all_B_sets, priors,
            unit_drop_edges, fixed_k, steps, burn_in, thin, 
            seed, 
            True, # 'progress' = False for workers
            candidate_pool, fitch_probs,
            a_step_inner_loops, z_step_inner_loops
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel chains (Clever Metropolis-within-Gibbs)...")
    
    with ProcessPoolExecutor(max_workers=min(n_chains, os.cpu_count() - 1)) as executor:
        # Submit all tasks
        futures = [executor.submit(_mcmc_worker_clever_mwg, t) for t in tasks]

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

        # futures = {executor.submit(_mcmc_worker_clever_mwg, t): i for i, t in enumerate(tasks)}

        # # Use tqdm to show progress as chains complete
        # results_iterator = as_completed(futures)
        # if 'trange' in globals(): # Check if tqdm is available
        #      results_iterator = trange(
        #          as_completed(futures), 
        #          total=n_chains, 
        #          desc="Running Clever MCMC Chains",
        #          leave=True
        #      )

        # for future in results_iterator:
        #     try:
        #         # Unpack results from a completed chain
        #         struct, score, stats = future.result()
        #         all_stats.append(stats)

        #         # Keep track of the single best solution found across all chains
        #         if score > best_score:
        #             best_score = score
        #             best_struct = struct

        #     except Exception as e:
        #         print(f"\n[ERROR] A clever MCMC chain failed with an error: {e}", file=sys.stderr)
        #         traceback.print_exc()

    print(f"[Info] All clever MCMC chains finished. Best score found: {best_score:.4f}")
    return best_struct, best_score, all_stats


def _mcmc_worker_A(args: tuple):
    """
    A simple worker function that unpacks arguments and calls the MCMC sampler.
    This is the target for each parallel process.
    """
    # Unpack all the arguments passed by the main parallel function
    (S, trees, leaf_type_maps, all_B_sets, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, candidate_pool, block_swap_sizes, Z) = args

    # Call the MCMC sampler with the unique seed for this worker
    # Note: We disable the progress bar for worker processes to keep the console clean
    return mcmc_map_search_only_A(
        S=S, trees=trees, leaf_type_maps=leaf_type_maps, all_B_sets=all_B_sets, priors=priors,
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
    (S, trees, leaf_type_maps, all_B_sets, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, candidate_pool, block_swap_sizes, fitch_probs) = args

    # Call the MCMC sampler with the unique seed for this worker
    # Note: We disable the progress bar for worker processes to keep the console clean
    return mcmc_map_search(
        S=S, trees=trees, leaf_type_maps=leaf_type_maps, all_B_sets=all_B_sets, priors=priors,
        unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
        burn_in=burn_in, thin=thin, seed=seed, progress=True,
        candidate_pool=candidate_pool, block_swap_sizes=block_swap_sizes, fitch_probs = fitch_probs
    )

def run_mcmc_only_A_parallel (
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ADD HERE
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
    Z
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
            S, trees, leaf_type_maps, all_B_sets, priors, unit_drop_edges, fixed_k,
            steps, burn_in, thin, seed, True, candidate_pool, block_swap_sizes, Z
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel MCMC chains...")
    with ProcessPoolExecutor(max_workers=min(n_chains, os.cpu_count() - 1)) as executor:
        futures = [executor.submit(_mcmc_worker_A, t) for t in tasks]

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
    

def run_mcmc_only_Z_parallel(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    all_B_sets: List[Dict[TreeNode, Set[str]]], # <--- ADD HERE
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
            S, trees, leaf_type_maps, all_B_sets, priors, unit_drop_edges, fixed_k,
            steps, burn_in, thin, seed, True, candidate_pool, block_swap_sizes, fitch_probs
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel MCMC chains...")
    with ProcessPoolExecutor(max_workers=min(n_chains, os.cpu_count() - 1)) as executor:
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

# ==============================================================================
# SECTION Z: MCMC Functions for Potency Sets (Z) using Collapsed Data
# ==============================================================================

# --- Function 1: Single-Chain MCMC for Z (adapted from _mcmc_worker_Z_collapsed) ---
def mcmc_map_search_only_Z_collapsed(
    S: List[str],
    collapsed_data: pd.DataFrame, # <-- Accepts collapsed_data
    priors: Priors,
    *,
    unit_drop_edges: bool = False, # Match Z-only assumption (A is fully connected conceptually)
    fixed_k: Optional[int] = None,
    steps: int = 500,
    burn_in: int = 100,
    thin: int = 10,
    seed: int = 123,
    progress: bool = True, # Controls progress bar display
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    block_swap_sizes: Tuple[int, ...] = (1,),
    fitch_probs: Optional[Dict[FrozenSet[str], float]] = None
) -> Tuple[Optional[Structure], float, Dict]:
    """ Single-chain MCMC sampler for Z using collapsed data. A is rebuilt deterministically. """

    rng_worker = random.Random(seed) # Use 'rng_worker' internally for consistency

    # --- Initialization Logic (copied/adapted from original _mcmc_worker_Z_collapsed) ---
    singles = {frozenset([t]) for t in S}
    root = frozenset(S)
    if candidate_pool is None: # Fallback if pool wasn't generated
         candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
         candidate_pool.append(root)
         candidate_pool = sorted(list(set(candidate_pool)), key=lambda x: (len(x), tuple(sorted(list(x)))))
    pool_set = set(candidate_pool) # Needed for inclusion calculation later

    # Initial Z state determination
    if priors.potency_mode == "fixed_k" and fixed_k is not None:
         Z0 = set(singles); Z0.add(root)
         available = [P for P in candidate_pool if P != root and P not in Z0]
         needed = max(0, fixed_k - 1)
         if len(available) < needed: raise RuntimeError(f"Not enough candidates ({len(available)}) to seed fixed_k={fixed_k}")
         Z0.update(rng_worker.sample(available, needed))
    else: # Bernoulli or unspecified k
        Z0 = set(singles); Z0.add(root)
        sprinkle = min(max(0, len([p for p in candidate_pool if p != root])//20), 5) # Sprinkle based on non-root pool
        available = [P for P in candidate_pool if P != root and P not in Z0]
        if sprinkle > 0 and len(available) >= sprinkle:
            Z0.update(rng_worker.sample(available, sprinkle))

    # Create initial structure using SubsetReach
    def make_struct_no_reach(Zset):
         A = _full_edges_for_Z(Zset, unit_drop_edges) # A is conceptually fully connected
         struct = Structure(S, Zset, A, unit_drop=unit_drop_edges)
         struct.Reach = SubsetReach() # IMPORTANT for Z-only scoring
         return struct

    current = make_struct_no_reach(Z0)
    current_logp_Z = priors.log_prior_Z(S, current.Z_active)
    # CALL COLLAPSED SCORER
    curr_score, _ = score_structure_no_edge_prior_collapsed(current, collapsed_data, priors, precomputed_logp_Z=current_logp_Z)

    # Retry initialization if needed
    retry_count = 0
    while not math.isfinite(curr_score) and retry_count < 20:
        if priors.potency_mode == "fixed_k" and fixed_k is not None:
             Z0 = set(singles); Z0.add(root)
             available = [P for P in candidate_pool if P != root and P not in Z0]
             needed = max(0, fixed_k - 1)
             if len(available) < needed: raise RuntimeError(f"Not enough candidates ({len(available)}) to seed fixed_k={fixed_k} on retry")
             Z0.update(rng_worker.sample(available, needed))
        else:
            Z0 = set(singles); Z0.add(root)
            sprinkle = min(5, len([p for p in candidate_pool if p != root]))
            available = [P for P in candidate_pool if P != root and P not in Z0]
            if sprinkle > 0 and len(available) >= sprinkle:
                 Z0.update(rng_worker.sample(available, sprinkle))
        current = make_struct_no_reach(Z0)
        current_logp_Z = priors.log_prior_Z(S, current.Z_active)
        # CALL COLLAPSED SCORER
        curr_score, _ = score_structure_no_edge_prior_collapsed(current, collapsed_data, priors, precomputed_logp_Z=current_logp_Z)
        retry_count += 1

    if not math.isfinite(curr_score):
        print(f"[ERROR] mcmc_map_search_only_Z (Seed {seed}): Could not find a finite-scoring starting point.", file=sys.stderr)
        return None, float('-inf'), {"error": "Initialization failed", "seed": seed, "scores": [], "all_scores_trace": []}


    # --- MCMC Loop (adapted from original _mcmc_worker_Z_collapsed) ---
    best_struct = current.clone()
    best_score = curr_score
    kept_Z = []
    kept_scores = []
    all_scores_trace = [curr_score]
    accepts = 0
    tried = 0

    # Define proposal function (fixed-k swap adapted slightly)
    def propose_fixed_k_swap_Z(Zset: Set[FrozenSet[str]]) -> Tuple[Optional[Set[FrozenSet[str]]], Optional[Dict]]:
        root = frozenset(S)
        acti = [P for P in Zset if len(P) >= 2 and P != root] # Active multis excluding root
        ina = [P for P in candidate_pool if P not in Zset]
        if not acti or not ina: return None, None
        # m = rng_worker.choice(block_swap_sizes) # Use local RNG
        m = 1 # Keep it simple, swap 1 based on previous fixes
        m = max(1, min(m, len(acti), len(ina)))

        # Weighted sampling (if fitch_probs provided)
        if fitch_probs:
            add_weights = [fitch_probs.get(p, 0.001) for p in ina]
            drop_weights = [1.0 - fitch_probs.get(p, 0.999) for p in acti]
            # Handle potential zero weights
            add = random.choices(ina, weights=add_weights, k=m) if sum(add_weights) > 1e-9 else rng_worker.sample(ina, m)
            drop = random.choices(acti, weights=drop_weights, k=m) if sum(drop_weights) > 1e-9 else rng_worker.sample(acti, m)
        else: # Uniform sampling
            add = rng_worker.sample(ina, m)
            drop = rng_worker.sample(acti, m)

        Z2 = set(Zset)
        Z2.difference_update(drop) # Use set ops for safety
        Z2.update(add)             # Use set ops for safety
        details = {'drop': drop, 'add': add}
        return Z2, details

    # Define proposal for Bernoulli mode (toggle)
    def propose_toggle_Z(Zset: Set[FrozenSet[str]]) -> Optional[Set[FrozenSet[str]]]:
         pool_multis = [p for p in candidate_pool if p != root] # Exclude root from toggling
         if not pool_multis: return None
         # Weighted toggle if fitch_probs available
         if fitch_probs:
             weights = [fitch_probs.get(p, 0.001) for p in pool_multis]
             # Handle potential zero weights
             P = random.choices(pool_multis, weights=weights, k=1)[0] if sum(weights) > 1e-9 else rng_worker.choice(pool_multis)
         else: # Uniform toggle
             P = rng_worker.choice(pool_multis)

         Z2 = set(Zset)
         if P in Z2: Z2.remove(P)
         else: Z2.add(P)
         return Z2

    # Iterator setup
    iterator = range(steps)
    # Use progress bar only if requested
    if progress:
        try:
             iterator = trange(steps, desc=f"MCMC-Z (Seed {seed})", leave=True) # Use leave=True for single chain
        except Exception:
             print(f"[Warning] Failed to initialize tqdm progress bar for Seed {seed}.", file=sys.stderr)
             iterator = range(steps) # Fallback
    counto = 0
    # --- MCMC ITERATION ---
    for it in iterator:
        prop_Z = None
        proposal_details = None

        # Choose proposal based on prior mode
        if priors.potency_mode == "fixed_k" and fixed_k is not None:
             prop_Z, proposal_details = propose_fixed_k_swap_Z(current.Z_active)
             proposed_logp_Z = current_logp_Z # Prior doesn't change for fixed-k swap
        else: # Bernoulli mode
             prop_Z = propose_toggle_Z(current.Z_active)
             if prop_Z is None: # No valid toggle possible
                 tried += 1 # Count attempt even if no proposal generated
                 continue
             # Recalculate prior for the proposed Z set
             proposed_logp_Z = priors.log_prior_Z(S, prop_Z)

        if prop_Z is None: # No proposal could be made (e.g., fixed_k swap failed)
            tried += 1
            continue

        prop_struct = make_struct_no_reach(prop_Z)
        # CALL COLLAPSED SCORER
        prop_score, _ = score_structure_no_edge_prior_collapsed(prop_struct, collapsed_data, priors, precomputed_logp_Z=proposed_logp_Z)

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            # Avoid math domain error for very negative delta
            accept_prob = math.exp(delta) if delta > -700 else 0.0 # Approx threshold for exp underflow
            accept = (delta >= 0) or (rng_worker.random() < accept_prob)

        tried += 1
        if accept:
            accepts += 1
            current = prop_struct
            curr_score = prop_score
            current_logp_Z = proposed_logp_Z # Update current prior score

            if curr_score > best_score:
                best_struct = current.clone() # Clone the accepted state
                best_score = curr_score
        counto = counto + 1
        all_scores_trace.append(curr_score) # Log score at every step
        # print(f"Iteration = {counto}, current score = {curr_score}")

        # Sample collection
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_Z.append({P for P in current.Z_active if len(P) >= 2})
            kept_scores.append(curr_score)

        # Update progress bar (if used)
        if progress and isinstance(iterator, tqdm): # Check if it's a tqdm object
            iterator.set_postfix({
                "logpost": f"{curr_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc": f"{(accepts/max(1,tried)):.2f}"
            }, refresh=True) # Use refresh=True

    # Close progress bar if used
    if progress and isinstance(iterator, tqdm):
        iterator.close()


    # Final stats calculation
    counts: Dict[FrozenSet[str], int] = defaultdict(int)
    # pool_set includes root and singletons, filter for inclusion calc
    pool_multis_set = {p for p in pool_set if len(p) >= 2}
    for Zs in kept_Z:
        for P in Zs:
            if P in pool_multis_set: # Only count candidates in the multi pool
                counts[P] += 1
    total_kept = max(1, len(kept_Z))
    # Calculate inclusion only over the multi-type candidate pool
    inclusion = {P: counts[P] / total_kept for P in pool_multis_set}

    stats = {
        "samples_Z": kept_Z,
        "scores": kept_scores,
        "all_scores_trace": all_scores_trace,
        "accept_rate": (accepts / max(1, tried)),
        "inclusion": inclusion,
        "seed": seed # Include seed for reference
    }
    # Ensure best_struct is returned, even if it's just the initial one
    if best_struct is None and math.isfinite(best_score):
         best_struct = current.clone() # Fallback if somehow best wasn't updated

    return best_struct, best_score, stats


# --- Function 2: Worker for Z-only MCMC (Calls Function 1) ---
def _mcmc_worker_Z_collapsed(args: tuple):
    """ Worker function for Z-only MCMC using collapsed data. """
    (S, collapsed_data, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, # 'progress' here is usually False for workers
     candidate_pool, block_swap_sizes, fitch_probs) = args

    # Call the single-chain function
    return mcmc_map_search_only_Z_collapsed(
        S=S, collapsed_data=collapsed_data, priors=priors,
        unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
        burn_in=burn_in, thin=thin, seed=seed, progress=progress, # Pass worker's progress flag
        candidate_pool=candidate_pool, block_swap_sizes=block_swap_sizes,
        fitch_probs=fitch_probs
    )


# --- Function 3: Parallel Runner for Z-only MCMC (Calls Function 2 via submit) ---
def run_mcmc_only_Z_parallel_collapsed(
    S: List[str],
    collapsed_data: pd.DataFrame, # <-- Takes DataFrame
    priors: Priors,
    *,
    unit_drop_edges: bool = False, # Match Z-only assumption
    fixed_k: Optional[int] = None,
    steps: int = 500,
    burn_in: int = 100,
    thin: int = 10,
    base_seed: int = 123,
    candidate_pool: Optional[List[FrozenSet[str]]] = None,
    block_swap_sizes: Tuple[int, ...] = (1,), # Default to 1 for Z-only
    n_chains: int = 4,
    fitch_probs: Optional[Dict[FrozenSet[str], float]] = None
):
    """ Runs multiple Z-only MCMC chains in parallel using collapsed data. """
    if n_chains <= 0:
        n_chains = max(1, os.cpu_count() - 1)


    seeds = [base_seed + i for i in range(n_chains)]

    # Package arguments for the worker function _mcmc_worker_Z_collapsed
    tasks = []
    for seed in seeds:
        tasks.append((
            S, collapsed_data, priors, unit_drop_edges, fixed_k,
            steps, burn_in, thin, seed, True, # Progress=False for workers
            candidate_pool, block_swap_sizes, fitch_probs
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel MCMC chains (Z-only search)...")

    
    with ProcessPoolExecutor(max_workers=min(n_chains, os.cpu_count() - 1)) as executor:
        # Submit the COLLAPSED worker function
        futures = {executor.submit(_mcmc_worker_Z_collapsed, t): i for i, t in enumerate(tasks)}

        for future in as_completed(futures): # Iterate using as_completed directly
            # chain_index = futures[future]
            try:
                # Unpack results
                struct, score, stats = future.result()
                all_stats.append(stats)
                # Track best
                if score > best_score: # Check score validity
                    best_score = score
                    best_struct = struct
            except Exception as e:
                print(f"A chain failed with an error: {e}")

    print(f"[Info] All chains finished. Best score found: {best_score:.4f}")
    return best_struct, best_score, all_stats

# ==============================================================================
# SECTION A: MCMC Functions for Edges (A) using Collapsed Data
# ==============================================================================

# --- Function 4: Single-Chain MCMC for A (adapted from _mcmc_worker_A_collapsed) ---
def mcmc_map_search_only_A_collapsed(
    S: List[str],
    collapsed_data: pd.DataFrame, # <-- Accepts collapsed_data
    priors: Priors,
    Z_fixed: Set[FrozenSet[str]], # <-- Requires fixed Z
    *,
    unit_drop_edges: bool = True,
    # fixed_k is needed for prior calculation, get from len(Z_fixed multis) or pass explicitly
    fixed_k: Optional[int] = None,
    steps: int = 600,
    burn_in: int = 150,
    thin: int = 10,
    seed: int = 456,
    progress: bool = True # Controls progress bar display
) -> Tuple[Optional[Structure], float, Dict]:
    """ Single-chain MCMC sampler for A using collapsed data and fixed Z. """

    rng_worker = random.Random(seed) # Use local RNG

    # --- Initialization ---
    # Calculate total admissible edges needed for fast prior calculation
    all_admissible_pairs = [
         (P, Q) for P in Z_fixed for Q in Z_fixed
         if admissible_edge(P, Q, unit_drop_edges)
    ]
    num_admissible_edges = len(all_admissible_pairs)

    # Determine fixed_k for prior calculation if not provided
    if fixed_k is None:
        fixed_k = sum(1 for P in Z_fixed if len(P) >= 2)

    # Calculate fixed log P(Z) once
    # Use a temporary prior object if mode needs overriding for this calc
    # prior_for_Z_calc = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=priors.rho)
    logp_Z_fixed = priors.log_prior_Z(S, Z_fixed)

    if not math.isfinite(logp_Z_fixed):
         print(f"[ERROR] mcmc_map_search_only_A (Seed {seed}): Provided Z_fixed has -inf prior (k={fixed_k}). Cannot proceed.", file=sys.stderr)
         return None, float('-inf'), {"error": f"Invalid Z prior for k={fixed_k}", "seed": seed, "scores": [], "all_scores_trace": []}


    # Seed A with a mid-sized connected DAG
    A0 = build_mid_sized_connected_dag(Z_fixed, keep_prob=0.35, unit_drop=unit_drop_edges, rng=rng_worker)
    current = Structure(S, Z_fixed, A0, unit_drop=unit_drop_edges)
    print("\nInitial Edges mid-size DAG:")
    edges = sorted([e for e, v in A0.items() if v == 1],
                   key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
    for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    # Initial scoring using the full collapsed score function
    # Use the main 'priors' object passed to the function here for P(A|Z)
    curr_score, _ = score_structure_collapsed(
         current, collapsed_data, priors, # Use original priors for P(A|Z)
         num_admissible_edges=num_admissible_edges,
         precomputed_logp_Z=logp_Z_fixed # Use the precalculated logP(Z)
    )

    # Retry initialization if score is invalid
    retry_count = 0
    while not math.isfinite(curr_score) and retry_count < 20:
         A0 = build_mid_sized_connected_dag(Z_fixed, keep_prob=0.35, unit_drop=unit_drop_edges, rng=rng_worker)
         current = Structure(S, Z_fixed, A0, unit_drop=unit_drop_edges)
         curr_score, _ = score_structure_collapsed(
              current, collapsed_data, priors,
              num_admissible_edges=num_admissible_edges,
              precomputed_logp_Z=logp_Z_fixed
         )
         retry_count += 1

    if not math.isfinite(curr_score):
        print(f"[ERROR] mcmc_map_search_only_A (Seed {seed}): Could not find a finite-scoring starting point.", file=sys.stderr)
        return None, float('-inf'), {"error": "Initialization failed", "seed": seed, "scores": [], "all_scores_trace": []}


    best_struct = current.clone()
    best_score = curr_score

    if not all_admissible_pairs:
        print("[Warning] No admissible edges found for the given Z_fixed set. MCMC will not move.")


    # --- MCMC Loop (adapted from original _mcmc_worker_A_collapsed) ---
    kept_A = [] # Store edge sets
    kept_scores = []
    all_scores_trace = [curr_score]
    kept_edge_density = []
    accepts = 0
    tried = 0

    # Proposal: Toggle an edge
    def prop_edge_toggle_A(st: Structure) -> Tuple[Optional[Structure], Optional[Dict]]:
         if not all_admissible_pairs: return None, None # No possible edges
         e = rng_worker.choice(all_admissible_pairs) # Use local RNG
         u, v = e
         new = st.clone()
         details = {'edge': e}
         if st.A.get(e, 0) == 1: # Edge exists, remove it
             del new.A[e]
             details['action'] = 'removed'
             # Use safe full recompute for removal
             new.recompute_reach() # Recomputes labels_list too if needed
         else: # Edge doesn't exist, add it
             new.A[e] = 1
             details['action'] = 'added'
             # Use incremental update for addition
             # Ensure labels_list is correct before calling incremental update
             if new.labels_list != st.labels_list: new.labels_list = new._sorted_labels()
             new.update_reach_add_edge(u, v)
         return new, details

    # Iterator setup
    iterator = range(steps)
    if progress:
        try:
             iterator = trange(steps, desc=f"MCMC-A (Seed {seed})", leave=True) # Use leave=True
        except Exception:
             print(f"[Warning] Failed to initialize tqdm progress bar for Seed {seed}.", file=sys.stderr)
             iterator = range(steps) # Fallback

    # --- MCMC ITERATION ---
    counto = 0

    for it in iterator:


        if not all_admissible_pairs:
            # If there are no possible edges, we can't do anything.
            break

        # 1. Propose a change by picking a random edge to flip
        edge_to_flip = rng_worker.choice(all_admissible_pairs)
        u, v = edge_to_flip
        
        action = 'added' if current.A.get(edge_to_flip, 0) == 0 else 'removed'
        last_proposal_details = {'edge': edge_to_flip, 'action': action}
        
        prop = current.clone()
        
        # Apply the toggle based on the action
        if action == 'added':
            # Use the fast, incremental update for additions.
            prop.A[edge_to_flip] = 1  # <-- YOU WERE MISSING THIS
            prop.update_reach_add_edge(u, v)
        else: # action == 'removed'
            # Use the safe, full re-computation for removals.
            del prop.A[edge_to_flip]
            # prop.A[edge_to_flip] = 0  # <-- YOU WERE MISSING THIS
            prop.update_reach_remove_edge(u, v)
        
        # prop.recompute_reach()

        # 2. Score the new structure using the parallel scorer
        prop_score, _ = score_structure_collapsed(
             prop, collapsed_data, priors, # Use original priors
             num_admissible_edges=num_admissible_edges,
             precomputed_logp_Z=logp_Z_fixed # Keep Z prior fixed
        )        

        accept = False
        if math.isfinite(prop_score):
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng_worker.random() < math.exp(delta))

        tried += 1
        if accept:
            accepts += 1

            if last_proposal_details and 'edge' in last_proposal_details:
                edge_details = last_proposal_details
                edge_str = f"{pot_str(edge_details['edge'][0])} -> {pot_str(edge_details['edge'][1])}"
                action_str = edge_details['action'].capitalize()
                print(f"[Accepted Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")

            current = prop
            curr_score = prop_score
            if curr_score >= best_score:
                best_struct = current.clone()
                best_score = curr_score
        
        else:
            if last_proposal_details and 'edge' in last_proposal_details:
                edge_details = last_proposal_details
                edge_str = f"{pot_str(edge_details['edge'][0])} -> {pot_str(edge_details['edge'][1])}"
                action_str = edge_details['action'].capitalize()
                print(f"[Rejected Edge]: {action_str} {edge_str} | New Score: {prop_score:.3f}")


        all_scores_trace.append(curr_score)
        
        # Sample collection
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_A.append(edges_from_A(current.A)) # Store the set of active edges
            kept_scores.append(curr_score)
            kept_edge_density.append(sum(v for v in current.A.values()))


        if progress and isinstance(iterator, tqdm): # Check if tqdm object
            iterator.set_postfix({
                "logpost": f"{curr_score:.3f}",
                "best": f"{best_score:.3f}",
                "acc": f"{(accepts/max(1,tried)):.2f}",
                "E": f"{len(current.A)}" # Number of edges
            }, refresh=True)

    # Close progress bar if used
    if progress and isinstance(iterator, tqdm):
        iterator.close()


    # Final Stats
    # Edge inclusion probability calculation
    edge_counts: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int] = defaultdict(int)
    for A_sample in kept_A:
        for edge in A_sample:
            # Only count admissible edges relevant to the fixed Z
            if edge[0] in Z_fixed and edge[1] in Z_fixed and edge in all_admissible_pairs:
                 edge_counts[edge] += 1
    total_kept = max(1, len(kept_A))
    # Calculate inclusion only over the set of admissible edges for this Z
    edge_inclusion = {edge: edge_counts.get(edge, 0) / total_kept for edge in all_admissible_pairs}


    stats = {
        "samples_A": kept_A, # List of edge sets
        "scores": kept_scores,
        "all_scores_trace": all_scores_trace,
        "edge_density": kept_edge_density,
        "accept_rate": (accepts / max(1, tried)),
        "edge_inclusion": edge_inclusion,
        "seed": seed
    }
    # Ensure best_struct is returned

    return best_struct, best_score, stats


# --- Function 5: Worker for A-only MCMC (Calls Function 4) ---
def _mcmc_worker_A_collapsed(args: tuple):
    """ Worker function for A-only MCMC using collapsed data. """
    (S, collapsed_data, priors, unit_drop_edges, fixed_k,
     steps, burn_in, thin, seed, progress, # 'progress' is usually False
     Z_fixed) = args # Z_fixed is passed

    # Call the single-chain function for A-only search
    return mcmc_map_search_only_A_collapsed(
        S=S, collapsed_data=collapsed_data, priors=priors, Z_fixed=Z_fixed,
        unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
        burn_in=burn_in, thin=thin, seed=seed, progress=progress # Pass worker's progress flag
    )


# --- Function 6: Parallel Runner for A-only MCMC (Calls Function 5 via submit) ---
def run_mcmc_only_A_parallel_collapsed(
    S: List[str],
    collapsed_data: pd.DataFrame, # <-- Takes DataFrame
    priors: Priors,
    *,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None, # Needed for prior
    steps: int = 600,
    burn_in: int = 150,
    thin: int = 10,
    base_seed: int = 456, # Different default seed
    n_chains: int = 4,
    Z: Set[FrozenSet[str]] # <-- Requires the fixed Z set
):
    """ Runs multiple A-only MCMC chains in parallel using collapsed data. """
    # --- Structure follows run_mcmc_only_Z_parallel_collapsed ---
    if n_chains <= 0:
        n_chains = max(1, os.cpu_count() - 1)

    seeds = [base_seed + i for i in range(n_chains)]

    # Package arguments for the worker _mcmc_worker_A_collapsed
    tasks = []
    for seed in seeds:
        tasks.append((
            S, collapsed_data, priors, unit_drop_edges, fixed_k,
            steps, burn_in, thin, seed, True, # Progress=False for workers
            Z # Pass the fixed Z set here
        ))

    best_struct = None
    best_score = float("-inf")
    all_stats = []

    print(f"[Info] Starting {n_chains} parallel MCMC chains (A-only search)...")

    with ProcessPoolExecutor(max_workers=min(n_chains, os.cpu_count() - 1)) as executor:
        # Submit the COLLAPSED A-worker function
        futures = {executor.submit(_mcmc_worker_A_collapsed, t) for t in tasks}

        # Use tqdm for the main process progress bar
        for future in as_completed(futures): # Iterate directly
            # chain_index = futures[future]
            try:
                # Unpack results
                struct, score, stats = future.result()
                all_stats.append(stats)
                # Track best
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

def score_given_map_and_trees(txt_path: str, trees, meta_paths, all_B_sets, fixed_k,
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

    _, per_tree_logs = score_structure(
        struct=struct,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        all_B_sets=all_B_sets,
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
    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    # raw_maps = [read_leaf_type_map_tls(p) for p in meta_paths]

    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]

    # --- MODIFICATION START ---

    # Build the set of all unique type strings
    all_types = {str(t) for m in leaf_type_maps for t in m.values()}
    
    # Discard the empty string '' if it exists in the set
    all_types.discard('') 
    
    # Note: If 'None' is also considered "empty", you can add:
    # all_types.discard('None')

    # Build the final sorted list S from the filtered set
    S = sorted(all_types)

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


# def process_case(map_idx: int, type_num: int, cells_n: int,
#                  priors, iters=100, restarts=5, log_dir: Optional[str]=None,
#                  tree_kind: str = "graph", n_jobs: Optional[int] = None):
#     # Resolve and validate all inputs (will print what it tries)
#     fate_map_path, idx4 = build_fate_map_path(map_idx, type_num, tree_kind=tree_kind)
#     tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n, tree_kind=tree_kind)

#     # load trees + maps
#     trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)

#     all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

#     ground_truth_sets, gt_loss, gt_Z_active, gt_edges= score_given_map_and_trees(
#         fate_map_path, trees, meta_paths, all_B_sets, fixed_k=priors.fixed_k
#     )

#     # run MAP search
#     # a compact candidate pool keeps things fast & well-mixed
#     print_fitch_potency_probs_once(
#         S, trees, leaf_type_maps,
#         header=f"\n[Potency ranking] type_{type_num}, map {idx4}, cells_{cells_n}"
#     )

#     pool = collect_fitch_multis(S, trees, leaf_type_maps)

#     # <<<--- CHANGE: Compute Fitch probabilities and convert to a dictionary --->>>
#     fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
#     fitch_probs_dict = {p: prob for p, prob in fitch_probs_list}

#     # NEW CALL
#     # The 'restarts' parameter now controls the number of parallel chains
#     bestF_Z, best_score_Z, all_stats_Z = run_mcmc_only_Z_parallel(
#         S=S,
#         trees=trees,
#         leaf_type_maps=leaf_type_maps,
#         all_B_sets=all_B_sets,  # <--- PASS IT HERE
#         priors=priors,
#         unit_drop_edges=False,
#         fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
#         steps=iters,
#         burn_in=(iters*15)//100,
#         thin=10,
#         base_seed=123,
#         candidate_pool=pool,
#         block_swap_sizes=(1, 2, 3),
#         n_chains= min(os.cpu_count() - 1, restarts),  # Use the 'restarts' argument to set the number of chains
#         fitch_probs = fitch_probs_dict
#     )

#     # --- START: ADDED CODE BLOCK ---
#     # Score the ground truth Z using the same Z-only scoring function for a fair comparison
#     # print("\n--- Scoring Ground Truth Z (Z-only search context) ---")
#     gt_logp_Z = priors.log_prior_Z(S, gt_Z_active)
#     if not math.isfinite(gt_logp_Z):
#         print("Ground Truth Z has -inf prior under the current model. Cannot score.")
#     else:
#         # Create a structure with GT potencies and a fully connected graph
#         A_full_gt = _full_edges_for_Z(gt_Z_active, unit_drop_edges=False)
#         gt_struct_for_Z_score = Structure(S, gt_Z_active, A_full_gt, unit_drop=False)
        
#         # IMPORTANT: Mimic the Z-only MCMC by using the SubsetReach object
#         gt_struct_for_Z_score.Reach = SubsetReach()
        
#         # Score using the no-edge-prior function
#         gt_score_Z_only, _ = score_structure_no_edge_prior(
#             gt_struct_for_Z_score, trees, leaf_type_maps, all_B_sets, priors, 
#             precomputed_logp_Z=gt_logp_Z
#         )
#         print(f"Score of Ground Truth Z (fully connected A): {gt_score_Z_only:.6f}")
#         print("----------------------------------------------------")
#     # --- END: ADDED CODE BLOCK ---

#     if all_stats_Z and log_dir:
#         plot_mcmc_traces(
#             all_stats=all_stats_Z,
#             # burn_in=(iters * 15) // 100,
#             # thin=10,
#             title=f"MCMC Trace for Potency Sets (Z) - Map {idx4}, Cells {cells_n}",
#             output_path=os.path.join(log_dir, f"trace_Z_type{type_num}_{idx4}_cells{cells_n}.png")
#         )


#     # Z_active = {
#     #     frozenset(['-11', '-12']),
#     #     frozenset(['-13', '-14']),
#     #     frozenset(['-2', '-7']),
#     #     frozenset(['-11', '-12', '-16']),
#     #     frozenset(['-13', '-14', '-16']),
#     #     frozenset(['-11', '-2', '-5', '-7']),
#     #     frozenset(['-11', '-2', '-4', '-5', '-7']),
#     #     frozenset(['-10', '-11', '-12', '-13', '-14', '-16']),
#     #     frozenset(['-10', '-11', '-12', '-13', '-14', '-16', '-2', '-4', '-5', '-7']),
#     #     frozenset(['-10']),
#     #     frozenset(['-11']),
#     #     frozenset(['-12']),
#     #     frozenset(['-13']),
#     #     frozenset(['-14']),
#     #     frozenset(['-16']),
#     #     frozenset(['-2']),
#     #     frozenset(['-4']),
#     #     frozenset(['-5']),
#     #     frozenset(['-7'])
#     # }

#     # iters = iters + 90

#     bestF, best_score, all_chain_stats = run_mcmc_only_A_parallel(
#         S=S,
#         trees=trees,
#         leaf_type_maps=leaf_type_maps,
#         all_B_sets=all_B_sets,  # <--- AND PASS IT HERE
#         priors=priors,
#         unit_drop_edges=False,
#         fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
#         steps=iters,
#         burn_in=(iters*15)//100,
#         thin=10,
#         base_seed=123,
#         candidate_pool=pool,
#         block_swap_sizes=(1, 2, 3),
#         n_chains= min(os.cpu_count() - 1, restarts),  # Use the 'restarts' argument to set the number of chains
#         Z = bestF_Z.Z_active
#         # Z = Z_active
#     )

#     if all_chain_stats and log_dir:
#         plot_mcmc_traces(
#             all_stats=all_chain_stats,
#             # burn_in=(iters * 15) // 100,
#             # thin=10,
#             title=f"MCMC Trace for Edges (A) - Map {idx4}, Cells {cells_n}",
#             output_path=os.path.join(log_dir, f"trace_A_type{type_num}_{idx4}_cells{cells_n}.png")
#         )

#     # For compatibility with your existing result processing,
#     # you can get the stats of the best chain if needed,
#     # or analyze all_chain_stats for convergence.
#     # For now, let's just get the acceptance rate from the first chain as an example.
#     stats = all_chain_stats[0] if all_chain_stats else {}

#     print("MCMC accept rate:", stats["accept_rate"])
#     # posterior inclusion frequency of each candidate potency:
#     incl = stats["inclusion"]

#     print(f"\n=== BEST MAP for type_{type_num}, map {idx4}, cells_{cells_n} ===")
#     multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
#                           key=lambda x: (len(x), tuple(sorted(list(x)))))
#     print("Active potencies (multi-type):")
#     for P in multi_sorted: print("  ", pot_str(P))
#     print("Singletons (always active):")
#     for t in S: print("  ", "{" + t + "}")

#     print("\nEdges:")
#     edges = sorted([e for e, v in bestF.A.items() if v == 1],
#                    key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
#     for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

#     print("\nScores:")
#     print(f"  log posterior: {best_score:.6f}")
#     # for i, lg in enumerate(per_tree_logs, 1):
#     #     print(f"  Tree {i} log P(T|F*): {lg:.6f}")

#     # --- Ground truth scoring ---
#     predicted_sets = {p for p in bestF.Z_active if len(p) > 1}

#     pretty_print_sets("Predicted Sets", predicted_sets)
#     pretty_print_sets("Ground Truth Sets", ground_truth_sets)

#     print("\n=== Ground Truth Directed Edges ===")
#     for (u, v) in sorted(gt_edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
#         print(f"{sorted(list(u))} -> {sorted(list(v))}")

#     jd = jaccard_distance(predicted_sets, ground_truth_sets)
#     print("\n=== Jaccard Distance ===")
#     print(f"Jaccard Distance (Pred vs GT): {jd:.6f}")
#     print(f"Predicted map's loss: {best_score:.6f}")
#     print(f"Ground truth's loss: {gt_loss:.6f}")

#     # --- EDGE METRICS (Predicted vs Ground Truth) ---
#     pred_edges = edges_from_A(bestF.A)
#     edge_jacc = jaccard_distance_edges(pred_edges, gt_edges)

#     nodes_union = gt_Z_active | set(bestF.Z_active)
#     im_d, im_s = ipsen_mikhailov_similarity(
#         nodes_union = nodes_union,
#         edges1 = pred_edges,
#         edges2 = gt_edges,
#         gamma = 0.08,
#     )

#     print("\n=== Edge-set Metrics ===")
#     print(f"Jaccard distance (edges): {edge_jacc:.6f}")
#     print(f"Ipsen–Mikhailov distance: {im_d:.6f}")
#     print(f"Ipsen–Mikhailov similarity: {im_s:.6f}")

#     # optional logs
#     # if log_dir:
#     #     os.makedirs(log_dir, exist_ok=True)
#     #     log_path = os.path.join(log_dir, f"log_type{type_num}_{idx4}_cells{cells_n}.txt")
#     #     with open(log_path, "w") as f:
#     #         f.write(f"type_{type_num}, map {idx4}, cells_{cells_n}\n")
#     #         f.write(f"Jaccard={jd:.6f}, GT loss={gt_loss:.6f}, Pred loss={best_score:.6f}\n")
#     return jd, gt_loss, best_score,edge_jacc,im_s

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
        fate_map_path, trees, meta_paths, all_B_sets, 
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
        all_B_sets=all_B_sets,
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
    final_logp_Z = priors.log_prior_Z(S, Z_map)
    final_num_admiss = len(A_full) # Num admissible edges for this Z
    final_logp_A = priors.log_prior_A(
        Z_map, A_map_final, unit_drop_edges, final_num_admiss
    )
    final_log_L = get_log_likelihood(
        bestF_final, trees, leaf_type_maps, all_B_sets
    )
    
    if not math.isfinite(final_logp_Z): final_logp_Z = 0 # Handle -inf prior
    if not math.isfinite(final_logp_A): final_logp_A = 0 # Handle -inf prior
    
    best_score_final = final_logp_Z + final_logp_A + final_log_L
    
    print(f"Final Log P(Z): {final_logp_Z:.4f}")
    print(f"Final Log P(A|Z): {final_logp_A:.4f} ({len(A_map_final)} edges)")
    print(f"Final Log L(Data|F): {final_log_L:.4f}")
    
    print(f"\n=== BEST MAP (Hybrid) for type_{type_num}, map {idx4}, cells_{cells_n} ===")
    multi_sorted = sorted([P for P in bestF_final.Z_active if len(P) >= 2],
                          key=lambda x: (len(x), tuple(sorted(list(x)))))
    print("Active potencies (multi-type):")
    for P in multi_sorted: print("  ", pot_str(P))
    
    print("\nEdges:")
    edges = sorted([e for e, v in bestF_final.A.items() if v == 1],
                   key=lambda e: (len(e[0]), tuple(sorted(list(e[0]))), pot_str(e[1])))
    for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  Log Posterior (Pred): {best_score_final:.6f}")
    print(f"  Log Posterior (GT):   {gt_loss:.6f}")

    # --- Potency Set Metrics ---
    predicted_sets = {p for p in bestF_final.Z_active if len(p) > 1}
    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", ground_truth_sets)
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

def process_case_clever_mwg(
    map_idx: int, 
    type_num: int, 
    cells_n: int,
    priors: "Priors", 
    iters: int, 
    n_chains: int,
    log_dir: Optional[str] = None,
    tree_kind: str = "graph",
    unit_drop_edges: bool = False, # Set default to False, as in your original
    a_steps: int = 5,
    z_steps: int = 1
):
    """
    Processes a single simulation case using the "Clever MwG" sampler.
    This replaces your original 'process_case' function.
    """
    
    # --- 1. Load Data and Ground Truth ---
    print(f"\n{'='*80}")
    print(f"=== Processing Case: type_{type_num}, map {map_idx:04d}, cells_{cells_n} ===")
    print(f"=== Sampler: Clever Metropolis-within-Gibbs (Joint Z+A) ===")
    print(f"=== k={priors.fixed_k}, iters={iters}, chains={n_chains}, A-steps={a_steps}, Z-steps={z_steps} ===")
    print(f"{'='*80}")

    fate_map_path, idx4 = build_fate_map_path(map_idx, type_num, tree_kind=tree_kind)
    tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n, tree_kind=tree_kind)

    trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
    all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

    ground_truth_sets, gt_loss, gt_Z_active, gt_edges = score_given_map_and_trees(
        fate_map_path, trees, meta_paths, all_B_sets, 
        fixed_k=priors.fixed_k,
        unit_drop_edges=unit_drop_edges
    )
    
    # --- 2. Prepare for MCMC ---
    print_fitch_potency_probs_once(
        S, trees, leaf_type_maps,
        header=f"\n[Potency ranking] type_{type_num}, map {idx4}, cells_{cells_n}"
    )
    pool = collect_fitch_multis(S, trees, leaf_type_maps)
    fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
    fitch_probs_dict = {p: prob for p, prob in fitch_probs_list}
    if not fitch_probs_dict: fitch_probs_dict = {}

    
    # --- 3. Run MCMC (Single Phase, Joint Sampler) ---
    
    bestF, best_score, all_stats = run_mcmc_clever_mwg_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        all_B_sets=all_B_sets,
        priors=priors,
        unit_drop_edges=unit_drop_edges,
        fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
        steps=iters,
        burn_in=(iters * 20) // 100,  # 20% burn-in
        thin=10,
        n_chains=n_chains,
        base_seed=123 + map_idx + cells_n, # Vary seed per case
        candidate_pool=pool,
        fitch_probs=fitch_probs_dict,
        a_step_inner_loops=a_steps,
        z_step_inner_loops=z_steps
    )

    if bestF is None:
        print("[ERROR] MCMC failed to find a valid structure for this case.")
        raise RuntimeError("MCMC failed")

    # --- 4. Plot Traces ---
    if all_stats and log_dir:
        plot_mcmc_traces(
            all_stats=all_stats,
            title=f"MCMC Trace (Clever MwG) - Map {idx4}, Cells {cells_n}, k={priors.fixed_k}",
            output_path=os.path.join(log_dir, f"trace_clever_type{type_num}_{idx4}_cells{cells_n}_k{priors.fixed_k}.png")
        )
        check_convergence(all_stats)

    # --- 5. Report Results & Metrics ---
    stats = all_stats[0] if all_stats else {}
    print("MCMC accept rates:", {
        "A_step": stats.get("accept_rate_A", "N/A"), 
        "Z_step": stats.get("accept_rate_Z", "N/A")
    })

    print(f"\n=== BEST MAP for type_{type_num}, map {idx4}, cells_{cells_n} ===")
    multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
                          key=lambda x: (len(x), tuple(sorted(list(x)))))
    print("Active potencies (multi-type):")
    for P in multi_sorted: print("  ", pot_str(P))
    
    print("\nEdges:")
    edges = sorted([e for e, v in bestF.A.items() if v == 1],
                   key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
    for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  Log Posterior (Pred): {best_score:.6f}")
    print(f"  Log Posterior (GT):   {gt_loss:.6f}") # From score_given_map_and_trees

    # --- Potency Set Metrics ---
    predicted_sets = {p for p in bestF.Z_active if len(p) > 1}
    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", ground_truth_sets)
    jd = jaccard_distance(predicted_sets, ground_truth_sets)
    print(f"Jaccard Distance (Potency Sets): {jd:.6f}")

    # --- Edge Set Metrics ---
    pred_edges = edges_from_A(bestF.A)
    edge_jacc = jaccard_distance_edges(pred_edges, gt_edges)
    
    nodes_union = gt_Z_active | set(bestF.Z_active)
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

    return jd, gt_loss, best_score, edge_jacc, im_s



def main_larry_collapsed(
    collapsed_mat_path: str, # Path to your collapsed_clone_type_mat.csv
    iters: int = 500,
    restarts: int = 4,
    fixed_k: int = 10,
    out_csv: str = "larry_collapsed_results.csv",
    log_dir: Optional[str] = "larry_collapsed_logs",
    unit_drop_edges: bool = False # Set based on your model assumption
):
    """ Main function to run MCMC analysis on collapsed LARRY data. """
    random.seed(42) # Base seed for consistency
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)

    try:
        # 1. Load collapsed matrix
        print(f"Loading collapsed potency matrix from: {collapsed_mat_path}")
        try:
            collapsed_data = pd.read_csv(collapsed_mat_path, index_col=0)
        except FileNotFoundError:
             print(f"Error: Input file not found at {collapsed_mat_path}", file=sys.stderr)
             return
        except Exception as e:
             print(f"Error reading CSV: {e}", file=sys.stderr)
             return

        # 2. Clean data
        if 'Undifferentiated' in collapsed_data.columns:
            print("Ignoring 'Undifferentiated' column.")
            collapsed_data = collapsed_data.drop(columns=['Undifferentiated'])
        if 'counts' not in collapsed_data.columns:
            raise ValueError("'counts' column missing")
        collapsed_data['counts'] = pd.to_numeric(collapsed_data['counts'], errors='coerce').fillna(0)
        collapsed_data['counts'] = collapsed_data['counts'].astype(int)
        collapsed_data = collapsed_data[collapsed_data['counts'] > 0]
        if collapsed_data.empty: raise ValueError("No valid data rows after cleaning.")

        # 3. Determine S
        S = sorted([col for col in collapsed_data.columns if col != 'counts'])
        if not S: raise ValueError("No cell type columns found.")
        print(f"Detected Type Universe S: {S}")

        # 4. Generate candidate pool from observed potencies + root
        pool = set()
        for idx, row in collapsed_data.iterrows():
             cell_types_present = {col for col in S if row[col] == 1}
             potency_set = frozenset(cell_types_present)
             if len(potency_set) >= 2: pool.add(potency_set)
        pool.add(frozenset(S)) # Ensure root is included
        candidate_pool = sorted(list(pool), key=lambda x: (len(x), tuple(sorted(list(x)))))
        print(f"Generated candidate pool of size {len(candidate_pool)} from input matrix.")

        # (Optional: Calculate Fitch probabilities - skipped for simplicity now)
        fitch_probs_dict = None

        # 5. MCMC Phase 1: Search for Best Z
        print("\n--- Starting MCMC Phase 1: Searching for Potency Sets (Z) ---")
        bestF_Z, best_score_Z, all_stats_Z = run_mcmc_only_Z_parallel_collapsed(
            S=S,
            collapsed_data=collapsed_data,
            priors=priors,
            unit_drop_edges=unit_drop_edges, # Use appropriate setting
            fixed_k=fixed_k,
            steps=iters,
            burn_in=(iters * 15) // 100,
            thin=10,
            base_seed=123,
            candidate_pool=candidate_pool,
            block_swap_sizes=(1,), # Swap 1 at a time
            n_chains=restarts,
            fitch_probs=fitch_probs_dict
        )

        if bestF_Z is None:
            print("[ERROR] MCMC Phase 1 (Z search) failed to find a valid structure.", file=sys.stderr)
            return

        print(f"--- MCMC Phase 1 Complete. Best Z score: {best_score_Z:.4f} ---")
        if all_stats_Z: check_convergence(all_stats_Z) # Check convergence

        # 6. MCMC Phase 2: Search for Best A given Best Z
        print("\n--- Starting MCMC Phase 2: Searching for Edges (A) given best Z ---")
        iters_A = iters # Slightly more iterations for A

        bestF_A, best_score_A, all_stats_A = run_mcmc_only_A_parallel_collapsed(
            S=S,
            collapsed_data=collapsed_data,
            priors=priors,
            unit_drop_edges=unit_drop_edges, # Use same setting
            fixed_k=fixed_k, # Pass k for prior calc
            steps=iters_A,
            burn_in=(iters_A * 15) // 100,
            thin=10,
            base_seed=456, # Different seed for A search
            n_chains=restarts,
            Z=bestF_Z.Z_active # Pass the best Z found
        )

        if bestF_A is None:
            print("[WARNING] MCMC Phase 2 (A search) failed. Using Z-search result structure.", file=sys.stderr)
            bestF = bestF_Z
            best_score = best_score_Z # Score corresponds to Z-only objective
        else:
            print(f"--- MCMC Phase 2 Complete. Best (Z,A) score: {best_score_A:.4f} ---")
            if all_stats_A: check_convergence(all_stats_A) # Check convergence
            bestF = bestF_A
            best_score = best_score_A # Score corresponds to Z+A objective

        # 7. Output Results
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if all_stats_Z:
                 plot_mcmc_traces(all_stats=all_stats_Z, title=f"MCMC Trace (Z search) - k={fixed_k}",
                                  output_path=os.path.join(log_dir, f"trace_Z_k{fixed_k}.png"))
            if all_stats_A:
                 plot_mcmc_traces(all_stats=all_stats_A, title=f"MCMC Trace (A search) - k={fixed_k}",
                                  output_path=os.path.join(log_dir, f"trace_A_k{fixed_k}.png"))

        print("\n=== FINAL BEST MAP STRUCTURE ===")
        multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
                              key=lambda x: (len(x), tuple(sorted(list(x)))))
        print("Active potencies (multi-type):")
        for P in multi_sorted: print("  ", pot_str(P))
        print("\nSingletons (always active):")
        for t in S: print("  ", "{" + t + "}")

        print("\nEdges:")
        edges = sorted([e for e, v in bestF.A.items() if v == 1],
                       key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
        for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

        print(f"\nFinal Log Posterior Score: {best_score:.6f}")

        # --- Save summary ---
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["InputFile", "FixedK", "BestLogPosterior", "NumMultiPotencies", "NumEdges"])
            writer.writerow([
                os.path.basename(collapsed_mat_path), fixed_k, f"{best_score:.6f}",
                len(multi_sorted), len(edges)
            ])
        print(f"\nResults summary saved to {out_csv}")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in main_larry_collapsed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

def main_tls(
    input_file_list: str, # Path to the file listing tree/meta pairs
    iters: int = 500,     # MCMC iterations (consider increasing)
    restarts: int = 4,    # Number of parallel chains
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
            all_B_sets=all_B_sets,
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
            fitch_probs=fitch_probs_dict
        )

        if bestF_Z is None:
            print("ERROR: MCMC Phase 1 (Z search) failed to find a valid structure.")
            return

        print(f"--- MCMC Phase 1 Complete. Best Z score: {best_score_Z:.4f} ---")
        # check_convergence(all_stats_Z) # Check convergence for Z search


        # --- MCMC Phase 2: Search for Best A given Best Z ---
        print("\n--- Starting MCMC Phase 2: Searching for Edges (A) given best Z ---")
        # Increase iterations slightly for edge search maybe
        iters_A = iters + 100
        # + max(50, iters // 10) # Example: add 10% or 50 steps

        bestF_A, best_score_A, all_stats_A = run_mcmc_only_A_parallel(
            S=S,
            trees=trees,
            leaf_type_maps=leaf_type_maps,
            all_B_sets=all_B_sets,
            priors=priors,
            unit_drop_edges=unit_drop_edges, # Use consistent setting
            fixed_k=fixed_k,
            steps=iters_A,
            burn_in=(iters_A * 15) // 100,
            thin=10,
            base_seed=456, # Use a different base seed
            candidate_pool=pool, # Not strictly needed but passed
            block_swap_sizes=(1,), # Not used in A-only search
            n_chains=restarts,
            Z=bestF_Z.Z_active # <<< Pass the best Z found in phase 1
        )

        if bestF_A is None:
            print("ERROR: MCMC Phase 2 (A search) failed. Using Z-search result.")
            bestF = bestF_Z # Fallback
            best_score = best_score_Z
        else:
             print(f"--- MCMC Phase 2 Complete. Best (Z,A) score: {best_score_A:.4f} ---")
            #  check_convergence(all_stats_A) # Check convergence for A search
             bestF = bestF_A
             best_score = best_score_A


        # --- Output Results ---
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # Plot traces if needed (using all_stats_Z and all_stats_A)
            if all_stats_Z:
                 plot_mcmc_traces(
                    all_stats=all_stats_Z, 
                    # burn_in=(iters * 15)//100, 
                    # thin=10,
                    title=f"MCMC Trace (Z search) - {os.path.basename(input_file_list)}",
                    output_path=os.path.join(log_dir, f"trace_Z_{os.path.basename(input_file_list)}.png")
                 )
            if all_stats_A:
                 plot_mcmc_traces(
                    all_stats=all_stats_A, 
                    # burn_in=(iters_A * 15)//100, 
                    # thin=10,
                    title=f"MCMC Trace (A search) - {os.path.basename(input_file_list)}",
                    output_path=os.path.join(log_dir, f"trace_A_{os.path.basename(input_file_list)}.png")
                 )


        print("\n=== FINAL BEST MAP STRUCTURE ===")
        multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
                              key=lambda x: (len(x), tuple(sorted(list(x)))))
        print("Active potencies (multi-type):")
        for P in multi_sorted: print("  ", pot_str(P))
        print("\nSingletons (always active):")
        for t in S: print("  ", "{" + t + "}")

        print("\nEdges:")
        edges = sorted([e for e, v in bestF.A.items() if v == 1],
                       key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
        for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

        print(f"\nFinal Log Posterior Score: {best_score:.6f}")

        # --- Save summary (optional) ---
        with open(out_csv, "w", newline="") as f:
             writer = csv.writer(f)
             writer.writerow(["InputFile", "FixedK", "BestLogPosterior", "NumMultiPotencies", "NumEdges"])
             writer.writerow([
                 input_file_list, fixed_k, f"{best_score:.6f}",
                 len(multi_sorted), len(edges)
             ])
        print(f"\nResults summary saved to {out_csv}")


    except FileNotFoundError as e:
        print(f"[ERROR] Input file not found: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred:")
        traceback.print_exc()

# def main_multi_type(type_nums=[10,14],
#                     maps_start=17, maps_end=26,
#                     cells_list=[50,100,200],
#                     iters = 50,
#                     restarts = 4,
#                     fixed_k = 5,
#                     out_csv="results_types_6_10_14_maps_17_26.csv",
#                     log_dir="logs_types",
#                     tree_kind: str = "graph"):
#     random.seed(7)
#     priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
#     results = []

#     with open(out_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Type","MapIdx","Cells","Jaccard","GT Loss","Pred Loss","Edge_Jaccard","IM_similarity"])

#         for t in type_nums:
#             for idx in range(maps_start, maps_end+1):
#                 for cells in cells_list:
#                     try:
#                         jd, gt_loss, pred_loss, edge_jacc, im_s = process_case(
#                             idx, t, cells, priors,
#                             iters=iters, restarts=restarts, log_dir=log_dir,
#                             tree_kind=tree_kind, n_jobs= os.cpu_count()-1  # start single-process
#                         )
#                         writer.writerow([t, idx, cells, f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}", f"{edge_jacc:.6f}", f"{im_s:.6f}"])
#                         results.append((t, idx, cells, jd, gt_loss, pred_loss, edge_jacc, im_s))
#                     except Exception as e:
#                         print(f"[WARN] Failed type_{t} map {idx:04d} cells_{cells}: {repr(e)}")
#                         traceback.print_exc()
#                         writer.writerow([t, idx, cells, "ERROR","ERROR","ERROR", "ERROR","ERROR"])
#                         results.append((t, idx, cells, None,None,None,None,None))


def main_multi_type( # This REPLACES your old `main_multi_type`
    type_nums=[10,14],
    maps_start=17, 
    maps_end=26,
    cells_list=[50,100,200],
    iters = 50,
    restarts = 4,
    fixed_k = 5,
    out_csv="results_viterbi_hybrid.csv",
    log_dir="logs_viterbi_hybrid",
    tree_kind: str = "graph"
):
    random.seed(7)
    # This must be False for the Viterbi approach, 
    # as F_full must contain all subset transitions.
    unit_drop_edges = False 
    
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
    results = []
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type","MapIdx","Cells","Jaccard","GT Loss","Pred Loss","Edge_Jaccard","IM_similarity"])

        for t in type_nums:
            for idx in range(maps_start, maps_end+1):
                for cells in cells_list:
                    try:
                        jd, gt_loss, pred_loss, edge_jacc, im_s = process_case(
                            idx, t, cells, priors,
                            iters=iters, 
                            restarts=restarts, 
                            log_dir=log_dir,
                            tree_kind=tree_kind, 
                            n_jobs= os.cpu_count()-1,
                            unit_drop_edges=unit_drop_edges # Pass this
                        )
                        writer.writerow([t, idx, cells, f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}", f"{edge_jacc:.6f}", f"{im_s:.6f}"])
                        results.append((t, idx, cells, jd, gt_loss, pred_loss, edge_jacc, im_s))
                    except Exception as e:
                        print(f"[WARN] Failed type_{t} map {idx:04d} cells_{cells}: {repr(e)}")
                        traceback.print_exc()
                        writer.writerow([t, idx, cells, "ERROR","ERROR","ERROR", "ERROR","ERROR"])
                        results.append((t, idx, cells, None,None,None,None,None))


def main_multi_type_clever_mwg(
    type_nums=[10,14],
    maps_start=17, 
    maps_end=26,
    cells_list=[50,100,200],
    iters=2000,
    n_chains=4,
    fixed_k=5,
    out_csv="results_clever_mcmc.csv",
    log_dir="logs_clever_mcmc",
    tree_kind: str = "graph",
    unit_drop_edges: bool = False,
    # New params
    a_steps: int = 5,
    z_steps: int = 1
):
    """
    Main batch processor. Calls 'process_case_clever_mwg' for each simulation.
    """
    random.seed(7)
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logs and plots will be saved to: {log_dir}")
        
    results = []

    # Write header for the new CSV
    csv_header = [
        "Type", "MapIdx", "Cells", "FixedK",
        "Jaccard_Sets", "GT_LogPost", "Pred_LogPost", 
        "Jaccard_Edges", "IM_Similarity",
        "Iters", "Chains", "A_Steps", "Z_Steps"
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for t in type_nums:
            for idx in range(maps_start, maps_end + 1):
                for cells in cells_list:
                    try:
                        jd, gt_loss, pred_loss, edge_jacc, im_s = process_case_clever_mwg(
                            map_idx=idx, 
                            type_num=t, 
                            cells_n=cells, 
                            priors=priors,
                            iters=iters, 
                            n_chains=n_chains, 
                            log_dir=log_dir,
                            tree_kind=tree_kind,
                            unit_drop_edges=unit_drop_edges,
                            a_steps=a_steps, 
                            z_steps=z_steps
                        )
                        
                        row = [
                            t, idx, cells, fixed_k,
                            f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}", 
                            f"{edge_jacc:.6f}", f"{im_s:.6f}",
                            iters, n_chains, a_steps, z_steps
                        ]
                        writer.writerow(row)
                        results.append(tuple(row)) # Save for summary

                    except Exception as e:
                        print(f"[WARN] FAILED CASE: type_{t} map {idx:04d} cells_{cells}: {repr(e)}")
                        traceback.print_exc()
                        row = [
                            t, idx, cells, fixed_k,
                            "ERROR", "ERROR", "ERROR", "ERROR", "ERROR",
                            iters, n_chains, a_steps, z_steps
                        ]
                        writer.writerow(row)
                        results.append(tuple(row))
                        
    print(f"\nBatch processing complete. Results saved to {out_csv}")


if __name__ == "__main__":

    # # --- Run the new "Clever MwG" sampler on your simulated data ---
    # main_multi_type_clever_mwg(
    #     type_nums=[6],
    #     maps_start=2,
    #     maps_end=2,
    #     cells_list=[50],
    #     iters = 100,           # Total MCMC steps per chain (low for testing)
    #     n_chains = 1,          # Number of parallel chains
    #     fixed_k = 5,
    #     out_csv="clever_mcmc_results.csv",
    #     log_dir="clever_mcmc_logs",
    #     tree_kind="graph",
    #     unit_drop_edges=False, # Use non-unit-drop edges
    #     a_steps=5,             # Inner loops for A-step
    #     z_steps=1              # Inner loops for Z-step
    # )

    #  # --- Main execution for collapsed LARRY data ---
    #  main_larry_collapsed(
    #      collapsed_mat_path="collapsed_clone_type_mat.csv", # Use the uploaded file name
    #      iters=100,      # Adjust MCMC iterations (e.g., 500-2000+)
    #      restarts=1,     # Adjust number of parallel chains (e.g., 4-8)
    #      fixed_k=7,      # << SET your desired number of multi-type potencies >>
    #      out_csv="larry_mcmc_results_k7.csv", # Output file name
    #      log_dir="larry_mcmc_logs_k7",    # Directory for trace plots
    #      unit_drop_edges=False # Set True if you want only single-step transitions
    #  )

     # You can add more calls to main_larry_collapsed here to test different k values
     # Example:
     # main_larry_collapsed(
     #     collapsed_mat_path="collapsed_clone_type_mat (1).csv",
     #     iters=500, restarts=4, fixed_k=8,
     #     out_csv="larry_mcmc_results_k8.csv", log_dir="larry_mcmc_logs_k8", unit_drop_edges=False
     # )


# if __name__ == "__main__":

#     #Only change in read_trees_and_maps

#     # --- NEW MAIN EXECUTION for TLS data ---
#     main_tls(
#         # **CHANGED**: Updated the input file path
#         input_file_list="TLS_locations.txt",
#         iters=100,
#         restarts=7,
#         fixed_k=7,
#         out_csv="tls_run_results_7_new2.csv",
#         log_dir="tls_run_logs"
#     )

# if __name__ == "__main__":
#     # --- Original main execution is commented out ---
#     # main_tls(
#     #     input_file_list="TLS_locations.txt",
#     #     iters=100,
#     #     restarts=7,
#     #     fixed_k=7,
#     #     out_csv="tls_run_results_7_new.csv",
#     #     log_dir="tls_run_logs"
#     # )

#     # <<< --- NEW CODE TO EVALUATE A SPECIFIC Z --- >>>

#     def evaluate_specific_Z(
#         input_file_list: str,
#         fixed_k_from_run: int, # The k used in the main run
#         multi_potencies_to_test: Set[FrozenSet[str]]
#     ):
#         """
#         Loads data and calculates the 'Z-only' MCMC score for a specific
#         set of multi-type potencies.
#         """
#         print(f"--- Evaluating Specific Z (original run k={fixed_k_from_run}) ---")
        
#         # This prior must match the one used in your main_tls run
#         priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k_from_run, rho=0.2)
#         unit_drop_edges = False # Match main_tls

#         try:
#             # 1. & 2. Load data (same as main_tls)
#             tree_paths, meta_paths = parse_input_file_list(input_file_list)
#             trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
#             print(f"Loaded {len(trees)} trees. Type universe S: {S}")

#             # 3. Precompute B_sets (same as main_tls)
#             all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

#             # 4. Construct the full Z_active set
#             singletons = {frozenset([t]) for t in S}
#             Z_active = multi_potencies_to_test | singletons

#             print("\nPotency sets being tested:")
#             for p in sorted(multi_potencies_to_test, key=lambda x: (len(x), tuple(sorted(x)))):
#                 print(f"  {pot_str(p)}")
            
#             # 5. Check k and calculate the log prior P(Z)
#             num_multis_provided = len(multi_potencies_to_test)
#             if num_multis_provided != fixed_k_from_run:
#                 print(f"\n[Warning] Number of provided sets ({num_multis_provided}) does not match the run's fixed_k ({fixed_k_from_run}).")
#                 print(f"[Info] The log P(Z) will be calculated using k={num_multis_provided} to match the provided set.")
#                 # Use a prior matching the *provided* number of sets for this evaluation
#                 priors_eval = Priors(potency_mode="fixed_k", fixed_k=num_multis_provided, rho=0.2)
#             else:
#                 print(f"\n[Info] Number of provided sets ({num_multis_provided}) matches run's fixed_k ({fixed_k_from_run}).")
#                 priors_eval = priors # Use the original prior

#             logp_Z = priors_eval.log_prior_Z(S, Z_active)
            
#             if not math.isfinite(logp_Z):
#                 print(f"ERROR: The provided Z set has a log prior of -inf.")
#                 print(f"This is likely because the number of available multi-type potencies")
#                 print(f"in the data (derived from S={S}) is less than k={priors_eval.fixed_k}.")
#                 return

#             print(f"Calculated log P(Z): {logp_Z:.6f}")

#             # 6. Create the special Structure for Z-only scoring
#             # The edge set A is irrelevant when using SubsetReach, so pass empty.
#             A_dummy = {} 
#             struct_to_score = Structure(S, Z_active, A_dummy, unit_drop=unit_drop_edges)
            
#             # CRITICAL STEP: Use SubsetReach to mimic the Z-only MCMC's
#             # assumption that reachability == subset.
#             struct_to_score.Reach = SubsetReach()

#             # 7. Score the structure using the Z-only-MCMC's score function
#             print("Scoring structure using 'score_structure_no_edge_prior'...")
#             z_only_score, _ = score_structure_no_edge_prior(
#                 struct_to_score,
#                 trees,
#                 leaf_type_maps,
#                 all_B_sets,
#                 priors_eval,
#                 precomputed_logp_Z=logp_Z
#             )

#             print("\n" + "="*30)
#             print(f"FINAL Z-ONLY SCORE: {z_only_score:.6f}")
#             print("="*30)

#         except FileNotFoundError as e:
#             print(f"[ERROR] Input file not found: {e}")
#         except Exception as e:
#             print(f"[ERROR] An unexpected error occurred:")
#             traceback.print_exc()

#     # --- Define the sets you want to test ---
#     # This is the list from your prompt
#     # target_multi_potencies = {
#     #     frozenset(['Endoderm','Endothelial','NMPs','NeuralTube','PCGLC','Somite']),
#     #     frozenset(['NMPs','NeuralTube']),
#     #     frozenset(['NeuralTube','Somite']),
#     #     frozenset(['Somite', 'Endothelial']),
#     #     frozenset(['Endothelial', 'NeuralTube','Somite']),
#     #     frozenset(['Endothelial', 'NeuralTube','PCGLC','Somite']),
#     #     frozenset(['NeuralTube','Somite','Endoderm']),
#     # }

#     # target_multi_potencies = {
#     #     frozenset(['NMPs','NeuralTube']),
#     #     frozenset(['NMPs','Somite']),
#     #     frozenset(['NeuralTube','Somite']),
#     #     frozenset(['NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endothelial','NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endoderm','Endothelial','NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endoderm','Endothelial','NMPs','NeuralTube','PCGLC','Somite'])
#     # }

#     # --- Run the evaluation ---
#     # These parameters must match your main_tls call
#     # evaluate_specific_Z(
#     #     input_file_list="TLS_locations.txt",
#     #     fixed_k_from_run=7, # This matches your main_tls call (fixed_k=7)
#     #     multi_potencies_to_test=target_multi_potencies
#     # )

#     # <<< --- FUNCTION 2: EVALUATE Z and A (NEW) --- >>>

#     def evaluate_specific_ZA(
#         title: str,
#         input_file_list: str,
#         fixed_k_from_run: int, # The k used in the main run
#         multi_potencies_to_test: Set[FrozenSet[str]],
#         edges_to_test: Set[Tuple[FrozenSet[str], FrozenSet[str]]]
#     ):
#         """
#         Loads data and calculates the 'full' MCMC score for a specific
#         set of multi-type potencies (Z) AND edges (A).
#         """
#         print(f"\n{'-'*60}\n--- {title} ---\n{'-'*60}")
#         print(f"Scoring Target: log P(Z) + log P(A|Z) + sum log P(T|Z,A) (Full MCMC score)")
        
#         # This prior must match the one used in your main_tls run
#         priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k_from_run, rho=0.2)
#         unit_drop_edges = False # Match main_tls

#         try:
#             # 1. & 2. Load data
#             tree_paths, meta_paths = parse_input_file_list(input_file_list)
#             trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
#             print(f"Loaded {len(trees)} trees. Type universe S: {S}")

#             # 3. Precompute B_sets
#             all_B_sets = [compute_B_sets(tree, ltm) for tree, ltm in zip(trees, leaf_type_maps)]

#             # 4. Construct the full Z_active set
#             singletons = {frozenset([t]) for t in S}
#             Z_active = multi_potencies_to_test | singletons

#             print("\nPotency sets (Z) being tested:")
#             for p in sorted(multi_potencies_to_test, key=lambda x: (len(x), tuple(sorted(x)))):
#                 print(f"  {pot_str(p)}")

#             # 5. Check k and calculate the log prior P(Z)
#             num_multis_provided = len(multi_potencies_to_test)
#             if num_multis_provided != fixed_k_from_run:
#                 print(f"\n[Warning] Number of provided Z sets ({num_multis_provided}) does not match run's fixed_k ({fixed_k_from_run}).")
#                 priors_eval = Priors(potency_mode="fixed_k", fixed_k=num_multis_provided, rho=0.2)
#             else:
#                 print(f"\n[Info] Number of provided Z sets ({num_multis_provided}) matches run's fixed_k ({fixed_k_from_run}).")
#                 priors_eval = priors

#             logp_Z = priors_eval.log_prior_Z(S, Z_active)
#             if not math.isfinite(logp_Z):
#                 print(f"ERROR: The provided Z set has a log prior of -inf.")
#                 return
#             print(f"Calculated log P(Z): {logp_Z:.6f}")

#             # 6. Construct the edge set (A)
#             A = {edge: 1 for edge in edges_to_test}
#             print(f"\nEdges (A) being tested ({len(A)} total):")
#             for u, v in sorted(A.keys(), key=lambda e: (len(e[0]), pot_str(e[0]), pot_str(e[1]))):
#                 print(f"  {pot_str(u)} -> {pot_str(v)}")

#             # 7. Calculate total number of *admissible* edges for P(A|Z)
#             num_admissible = 0
#             for P in Z_active:
#                 for Q in Z_active:
#                     if admissible_edge(P, Q, unit_drop_edges):
#                         num_admissible += 1
#             print(f"\nTotal admissible edges for this Z: {num_admissible}")
            
#             # 8. Create the full Structure object
#             # This will compute the real transitive closure (Reach)
#             print("Building full structure (with transitive closure)...")
#             struct_to_score = Structure(S, Z_active, A, unit_drop=unit_drop_edges)

#             # 9. Score the structure using the FULL score function
#             print("Scoring structure using 'score_structure'...")
#             full_score, _ = score_structure(
#                 struct_to_score,
#                 trees,
#                 leaf_type_maps,
#                 all_B_sets,
#                 priors_eval,
#                 num_admissible_edges=num_admissible, # Pass this for fast P(A|Z)
#                 precomputed_logp_Z=logp_Z
#             )

#             print("\n" + "="*30)
#             print(f"FINAL FULL SCORE: {full_score:.6f}")
#             print("="*30)

#         except FileNotFoundError as e:
#             print(f"[ERROR] Input file not found: {e}")
#         except Exception as e:
#             print(f"[ERROR] An unexpected error occurred:")
#             traceback.print_exc()

#     # =================================================================
#     # --- Run Evaluations ---
#     # =================================================================

#     # --- Evaluation 1: The Z-set from your previous prompt ---
    
#     # This is the list from your last prompt
#     # prev_target_multi_potencies = {
#     #     frozenset(['NMPs','NeuralTube']),
#     #     frozenset(['NMPs','Somite']),
#     #     frozenset(['NeuralTube','Somite']),
#     #     frozenset(['NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endothelial','NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endoderm','Endothelial','NMPs','NeuralTube','Somite']),
#     #     frozenset(['Endoderm','Endothelial','NMPs','NeuralTube','PCGLC','Somite'])
#     # }
    
#     # evaluate_specific_Z(
#     #     title="Evaluation 1: User-Provided Z set (Z-only score)",
#     #     input_file_list="TLS_locations.txt",
#     #     fixed_k_from_run=7, # This matches your main_tls call (fixed_k=7)
#     #     multi_potencies_to_test=prev_target_multi_potencies
#     # )

#     # --- Evaluation 2: The Z and A from the CARTA GRAPH IMAGE ---

#     # Define potencies (Z) from the Carta graph
#     # P_root = frozenset(['NMPs', 'NeuralTube', 'Somite', 'Endoderm', 'Endothelial', 'PCGLC'])
#     # P_MNS = frozenset(['NMPs', 'NeuralTube', 'Somite'])
#     # P_NSD = frozenset(['NeuralTube', 'Somite', 'Endoderm'])
#     # P_NSTP = frozenset(['NeuralTube', 'Somite', 'Endothelial', 'PCGLC'])
#     # P_MN = frozenset(['NMPs', 'NeuralTube'])
#     # P_NS = frozenset(['NeuralTube', 'Somite'])
#     # P_ST = frozenset(['Somite', 'Endothelial'])

#     P_NN = frozenset(['NMPs', 'NeuralTube'])
#     P_NS = frozenset(['NMPs', 'Somite'])
#     P_TS = frozenset(['NeuralTube', 'Somite'])
#     P_NNS = frozenset(['NMPs', 'NeuralTube', 'Somite'])
#     P_ENNS = frozenset(['Endothelial', 'NMPs', 'NeuralTube', 'Somite'])
#     P_DENNS = frozenset(['Endoderm', 'Endothelial', 'NMPs', 'NeuralTube', 'Somite'])
#     P_DENNPS = frozenset(['Endoderm', 'Endothelial', 'NMPs', 'NeuralTube', 'PCGLC', 'Somite'])
    
#     # graph_multi_potencies = {
#     #     P_root, P_MNS, P_NSD, P_NSTP, P_MN, P_NS, P_ST
#     # } # This is 7 sets, matching k=7

#     new_graph_multi_potencies = {
#         P_NN, P_NS, P_TS, P_NNS, P_ENNS, P_DENNS, P_DENNPS
#     } # This is 7 sets, matching k=7

#     # Define singletons (for edge definitions) using DATA strings
#     P_M = frozenset(['NMPs'])
#     P_N = frozenset(['NeuralTube'])
#     P_S = frozenset(['Somite'])
#     P_D = frozenset(['Endoderm'])
#     P_T = frozenset(['Endothelial'])
#     P_P = frozenset(['PCGLC'])

#     # Define the 16 edges (A) from your new list
#     new_graph_edges = {
#         (P_NN, P_N),
#         (P_NS, P_M),
#         (P_TS, P_N),
#         (P_NNS, P_N),
#         (P_NNS, P_S),
#         (P_NNS, P_NS),
#         (P_ENNS, P_NNS),
#         (P_DENNS, P_D),
#         (P_DENNS, P_T),
#         (P_DENNS, P_NS),
#         (P_DENNS, P_TS),
#         (P_DENNS, P_NNS),
#         (P_DENNS, P_ENNS),
#         (P_DENNPS, P_NN),
#         (P_DENNPS, P_NS),
#         (P_DENNPS, P_DENNS),
#     } # This is 16 edges


#     # Define edges (A) from the Carta graph (excluding non-subset edges)
#     # graph_edges = {
#     #     (P_root, P_MNS),
#     #     (P_root, P_NSD),
#     #     (P_root, P_NSTP),
#     #     (P_MNS, P_MN),
#     #     (P_MNS, P_NS),
#     #     (P_MNS, P_S),  # Red edge M,N,S -> S
#     #     (P_MNS, P_N),
#     #     (P_NSD, P_NS),
#     #     (P_NSD, P_D),
#     #     (P_NSD, P_S),
#     #     (P_NSTP, P_ST),
#     #     (P_NSTP, P_NS),
#     #     (P_NSTP, P_S), # Red edge N,S,T,P -> S
#     #     (P_NSTP, P_P),
#     #     (P_MN, P_M),
#     #     (P_MN, P_N),
#     #     (P_NS, P_N),
#     #     (P_NS, P_S),
#     #     (P_ST, P_S),
#     #     (P_ST, P_T)
#     # } # This is 20 edges

#     evaluate_specific_ZA(
#         title="Evaluation 2: Carta Graph Structure (Full Z+A score)",
#         input_file_list="TLS_locations.txt",
#         fixed_k_from_run=7, # This matches your main_tls call (fixed_k=7)
#         multi_potencies_to_test=new_graph_multi_potencies,
#         edges_to_test=new_graph_edges
#     )


    main_multi_type(
        type_nums=[6],
        maps_start=2,
        maps_end=6,
        cells_list=[50],
        iters = 100,
        restarts = 7,
        fixed_k = 5,
        out_csv="viterbi_6_50.csv",
        log_dir="viterbi_6_50",
        tree_kind="graph"   # or "bin_trees" or "graph"
    )