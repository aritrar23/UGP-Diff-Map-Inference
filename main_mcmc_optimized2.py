from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import csv
import json
import math, random
import itertools
from typing import Iterable,  Dict, Tuple, List, Optional, Set, FrozenSet
from collections import Counter, defaultdict
from tqdm import trange
import traceback
import numpy as np
import numba
# import matplotlib
# matplotlib.use('Agg')  # MUST BE CALLED BEFORE importing pyplot
# import matplotlib.pyplot as plt
import arviz as az
# Add this with your other imports
from scipy.special import gammaln # This is the vectorized log-gamma function


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

# def get_structure_key(struct: Structure) -> tuple:
#     """Creates a hashable, unique key from a Structure object."""
#     # frozenset is an immutable, hashable set.
#     z_key = frozenset(struct.Z_active)
#     a_key = frozenset(struct.A.items())
#     return (z_key, a_key)

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
def assign_union_potency(root: TreeNode, leaf_type_map: Dict[str, str]) -> Set[str]:
    """
    Post-order union-only labeling. Sets `node.potency` for every node (as a Python set).
    For leaves, looks up leaf_type_map[node.name] to get the leaf cell type.
    Returns the potency set at `root`.
    """
    if root.is_leaf():
        if root.name is None:
            raise KeyError("Leaf has no .name; cannot map to leaf_type_map")
        if root.name not in leaf_type_map:
            raise KeyError(f"Leaf name '{root.name}' not found in leaf_type_map")
        root.potency = {leaf_type_map[root.name]}
        return root.potency

    union_set: Set[str] = set()
    for child in root.children:
        child_set = assign_union_potency(child, leaf_type_map)
        union_set |= child_set
    root.potency = union_set
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


import numpy as np

def transitive_closure_numpy(
    labels: List[FrozenSet[str]],
    A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]
) -> Dict[FrozenSet[str], Set[FrozenSet[str]]]:
    """
    Computes the transitive closure of a directed graph using NumPy for performance.
    """
    idx = {L: i for i, L in enumerate(labels)}
    n = len(labels)
    
    # 1. Create a boolean adjacency matrix
    M = np.zeros((n, n), dtype=bool)
    
    # 2. A node can always reach itself (reflexive)
    np.fill_diagonal(M, True)

    # 3. Add the direct edges from the adjacency dictionary A
    for (P, Q), v in A.items():
        if v == 1:
            i, j = idx.get(P), idx.get(Q)
            # Ensure both potencies are in the current label set
            if i is not None and j is not None:
                M[i, j] = True

    # 4. Use a vectorized Floyd-Warshall-style algorithm.
    # This loop is significantly faster in NumPy than as nested Python loops.
    for k in range(n):
        # For every pair of nodes (i, j), the new reachability M[i, j] is
        # the old reachability OR (is there a path from i to k AND a path from k to j).
        M_col_k = M[:, k, np.newaxis] # Path from i to k
        M_row_k = M[np.newaxis, k, :] # Path from k to j
        M |= (M_col_k & M_row_k)

    # 5. Convert the final matrix back to the dictionary format required by the DP.
    Reach = {}
    for i, L in enumerate(labels):
        # np.where is a fast way to get the indices of all reachable nodes
        reachable_indices = np.where(M[i, :])[0]
        Reach[L] = {labels[j] for j in reachable_indices}
        
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

# The NEW signature accepts max_dim
def dp_tree_root_table_numpy(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
    max_dim: int,
    prune_eps: float = 0.0
) -> np.ndarray:
    
    label_index = {L: i for i, L in enumerate(active_labels)}
    memo: Dict[Tuple[int, int], np.ndarray] = {}
    def nid(v: TreeNode) -> int: return id(v)

    def M(v: TreeNode, P: Optional[FrozenSet[str]]) -> np.ndarray:
        key = (nid(v), -1 if P is None else label_index[P])
        if key in memo:
            return memo[key]

        if v.is_leaf():
            res = np.array([[0.0]], dtype=np.float64)
            memo[key] = res
            return res

        Bv = B_sets.get(v, set())
        out_table = np.full((1, 1), -np.inf, dtype=np.float64)

        parent_reach = active_labels if P is None else list(Reach.get(P, []))

        for L in parent_reach:
            if not Bv.issubset(L):
                continue
            
            o_local = len(L & Bv)
            d_local = len(L - Bv)

            child_tabs = [M(u, L) for u in v.children]
            if any(tab.size == 0 for tab in child_tabs):
                continue
            
            convolved_tab = np.array([[0.0]], dtype=np.float64)
            if child_tabs:
                convolved_tab = child_tabs[0]
                for t in child_tabs[1:]:
                    convolved_tab = convolve_2d_log_numba(convolved_tab, t)
            
            h, w = convolved_tab.shape
            required_h = o_local + h
            required_w = d_local + w
            current_h, current_w = out_table.shape

            if required_h > current_h or required_w > current_w:
                new_h = max(current_h, required_h)
                new_w = max(current_w, required_w)
                new_out_table = np.full((new_h, new_w), -np.inf, dtype=np.float64)
                new_out_table[:current_h, :current_w] = out_table
                out_table = new_out_table
            
            target_slice_o = slice(o_local, o_local + h)
            target_slice_d = slice(d_local, d_local + w)
            
            # <<< NEW: Add an assertion for robust debugging >>>
            # This will give a clear error if the shapes don't match for any reason.
            target_shape = out_table[target_slice_o, target_slice_d].shape
            assert target_shape == convolved_tab.shape, \
                f"Shape mismatch! Target slice is {target_shape}, but convolved table is {convolved_tab.shape}."

            out_table[target_slice_o, target_slice_d] = np.logaddexp(
                out_table[target_slice_o, target_slice_d],
                convolved_tab
            )

        valid_rows = np.where(np.any(out_table > -np.inf, axis=1))[0]
        valid_cols = np.where(np.any(out_table > -np.inf, axis=0))[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            final_table = out_table[:valid_rows[-1]+1, :valid_cols[-1]+1]
        else:
            final_table = np.array([[]], dtype=np.float64)

        memo[key] = final_table
        return final_table

    return M(root, None)

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
    
# A Numba-JITed version of logsumexp for scalar values.
@numba.njit
def logsumexp_numba(a, b):
    """Numerically stable log(exp(a) + exp(b)) compiled by Numba."""
    if a == -np.inf: return b
    if b == -np.inf: return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

# The core replacement for your dictionary-based convolution.
# This is the powerhouse of the optimization.
@numba.njit
def convolve_2d_log_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Fast convolution of 2D log-space DP tables using Numba.
    A and B are 2D NumPy arrays where values are log-probabilities.
    """
    # Determine the shape of the resulting convolved array.
    # The new dimension is (dimA + dimB - 1).
    out_shape = (A.shape[0] + B.shape[0] - 1, A.shape[1] + B.shape[1] - 1)
    
    # Initialize the output array with -np.inf, which is log(0).
    out = np.full(out_shape, -np.inf, dtype=np.float64)

    # Perform the convolution with explicit loops.
    # Numba will compile these loops into highly optimized machine code.
    for o1 in range(A.shape[0]):
        for d1 in range(A.shape[1]):
            log_w1 = A[o1, d1]
            if log_w1 == -np.inf:
                continue  # Skip if the probability is zero.
            
            for o2 in range(B.shape[0]):
                for d2 in range(B.shape[1]):
                    log_w2 = B[o2, d2]
                    if log_w2 == -np.inf:
                        continue
                    
                    # The new coordinates are the sum of the old ones.
                    o_new, d_new = o1 + o2, d1 + d2
                    # In log-space, multiplication becomes addition.
                    log_w_new = log_w1 + log_w2
                    
                    # Accumulate probabilities using logsumexp.
                    out[o_new, d_new] = logsumexp_numba(out[o_new, d_new], log_w_new)
    return out

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

def tree_marginal_from_root_table_log_numpy(C_log: np.ndarray) -> float:
    """
    Calculates the final log marginal probability from a log-space DP NumPy array.
    (Fully vectorized version)
    """
    if C_log.size == 0 or np.all(C_log == -np.inf):
        return -math.inf

    # Find indices (O, D) where the log-probability is not -inf
    O_indices, D_indices = np.where(C_log > -np.inf)
    
    if O_indices.size == 0:
        return -math.inf

    log_weights = C_log[O_indices, D_indices]

    # <<< NEW: Fully vectorized log-beta calculation using scipy.special.gammaln >>>
    # This is faster and more robust than the previous list-comprehension method.
    log_beta_terms = (
        gammaln(O_indices + 1) + 
        gammaln(D_indices + 1) - 
        gammaln(O_indices + D_indices + 2)
    )
    
    total_log_terms = log_weights + log_beta_terms

    # Use a stable log-sum-exp to get the final marginal log-likelihood
    max_log = np.max(total_log_terms)
    if max_log == -np.inf:
        return -np.inf
        
    logL = max_log + np.log(np.sum(np.exp(total_log_terms - max_log)))
    
    return float(logL)
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

    # <<< MODIFIED: This now accepts the pre-computed list of edges >>>
    def log_prior_A(self,
                    A:Dict[Tuple[FrozenSet[str],FrozenSet[str]],int],
                    all_admissible_edges: List[Tuple[FrozenSet[str], FrozenSet[str]]]
                   )->float:
        logp=0.0
        # Instead of two nested loops, we iterate over the smaller list of valid edges.
        for P, Q in all_admissible_edges:
            # a == 1 if the edge is present in A, else 0
            a = 1 if A.get((P,Q), 0) == 1 else 0
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
                 A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int],
                 unit_drop: bool = True):
        self.S = S
        self.Z_active = set(Z_active)
        self.A = dict(A)
        self.unit_drop = unit_drop

        # <<< FIX: This line MUST come first >>>
        # 1. Create self.labels_list.
        self.labels_list = self._sorted_labels()
        
        # <<< This line is now correct because self.labels_list exists >>>
        # 2. Now, compute admissible edges.
        self.all_admissible_edges = self._compute_admissible_edges()
        
        # <<< This line is also now correct >>>
        # 3. Finally, compute reachability.
        self.Reach = transitive_closure_numpy(self.labels_list, self.A)

    def _sorted_labels(self)->List[FrozenSet[str]]:
        # Provide a stable, human-logical ordering of the active labels:
        #   1) by set size (|L|), then
        #   2) lexicographically by the sorted elements of the set.
        # This keeps DP indices stable and makes printed output neat.
        return sorted(list(self.Z_active), key=lambda x: (len(x), tuple(sorted(list(x)))))
    
      # <<< ADDED: New method to compute the edges once >>>
    def _compute_admissible_edges(self) -> List[Tuple[FrozenSet[str], FrozenSet[str]]]:
        """Generates all valid P -> Q edges for the current Z_active set."""
        pairs = []
        L = self.labels_list # Use the sorted list for consistency
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, self.unit_drop):
                    pairs.append((P, Q))
        return pairs

    def recompute_reach(self):
        self.labels_list=self._sorted_labels()
        
        # <<< ADDED: Re-compute the admissible edges when Z changes >>>
        self.all_admissible_edges = self._compute_admissible_edges()
        
        self.Reach = transitive_closure_numpy(self.labels_list, self.A)

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


    # <<< MODIFIED: This method now uses the pre-computed list >>>
    def propose_add_edge(self, rng:random.Random)->Optional["Structure"]:
        # Use the pre-computed list for a faster lookup of candidate edges.
        pairs = [e for e in self.all_admissible_edges if self.A.get(e,0)==0]
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
        
# ----------------------------
# Scoring: log posterior
# ----------------------------

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
    
    # <<< MODIFIED: Pass the pre-computed edge list to the prior function >>>
    logp += priors.log_prior_A(struct.A, struct.all_admissible_edges)

    # ---- Likelihood over all trees ----
    # ... (the rest of the function is unchanged, it uses the NumPy DP) ...
    logLs = []
    max_dim = len(struct.S)
    for root, leaf_to_type in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_to_type)
        if not B_sets.get(root):
            logLs.append(0.0)
            continue

        C_log_np = dp_tree_root_table_numpy(
            root, struct.labels_list, struct.Reach, B_sets, max_dim, prune_eps=prune_eps
        )
        tree_logL = tree_marginal_from_root_table_log_numpy(C_log_np)

        if not math.isfinite(tree_logL):
            return float("-inf"), []

        logLs.append(tree_logL + logp)

    return sum(logLs), logLs

def score_structure_no_edge_prior(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    priors: Priors,
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # Compute the (log) posterior score of a candidate structure F = (Z_active, A).

    # ---- Prior over structure F ----
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp):
        return float("-inf"), []
    

    # ---- Likelihood over all trees ----
    # ... (the rest of the function is unchanged, it uses the NumPy DP) ...
    logLs = []
    max_dim = len(struct.S)
    for root, leaf_to_type in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_to_type)
        if not B_sets.get(root):
            logLs.append(0.0)
            continue

        C_log_np = dp_tree_root_table_numpy(
            root, struct.labels_list, struct.Reach, B_sets, max_dim, prune_eps=prune_eps
        )
        tree_logL = tree_marginal_from_root_table_log_numpy(C_log_np)

        if not math.isfinite(tree_logL):
            return float("-inf"), []

        logLs.append(tree_logL + logp)

    return sum(logLs), logLs

# def score_structure(struct: Structure,
#                     trees: List[TreeNode],
#                     leaf_type_maps: List[Dict[str,str]],
#                     priors: Priors,
#                     prune_eps: float = 0.0) -> Tuple[float, List[float]]:
#     # Compute the (log) posterior score of a candidate structure F = (Z_active, A).

#     # ---- Prior over structure F ----
#     logp = priors.log_prior_Z(struct.S, struct.Z_active)
#     if not math.isfinite(logp):
#         return float("-inf"), []
#     # Add the edge prior, which is the key difference in this function
#     logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

#     # ---- Likelihood over all trees ----
#     logLs = []
#     for root, leaf_to_type in zip(trees, leaf_type_maps):
#         B_sets = compute_B_sets(root, leaf_to_type)
#         root_labels = B_sets.get(root, set())

#         if not root_labels:
#             logLs.append(0.0)
#             continue

#         # CORRECTED SECTION: Use the log-space DP and marginalization functions
#         # 1. Call the DP function that returns a log-space table.
#         C_log = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)

#         if not C_log:
#             # If the DP table is empty, this structure is impossible for this tree.
#             return float("-inf"), []

#         # 2. Call the marginalization function that works entirely in log-space.
#         tree_logL = tree_marginal_from_root_table_log(C_log)

#         if not math.isfinite(tree_logL):
#             # If the final log-likelihood is not a valid number, abort.
#             return float("-inf"), []

#         # 3. Append the per-tree log-likelihood directly. No need for math.log().
#         logLs.append(tree_logL + logp)

#     # Total posterior score = log prior + sum of per-tree log-likelihoods
#     total_log_post = sum(logLs)
#     return total_log_post, logLs


# def score_structure_no_edge_prior(struct: Structure,
#                     trees: List[TreeNode],
#                     leaf_type_maps: List[Dict[str,str]],
#                     priors: Priors,
#                     prune_eps: float = 0.0) -> Tuple[float, List[float]]:
#     # print("\n" + "="*50)
#     # print("=== STARTING SCORE_STRUCTURE ===")
    
#     logp = priors.log_prior_Z(struct.S, struct.Z_active)
#     # print(f"[DEBUG score_structure] Log Prior P(Z): {logp}")
#     if not math.isfinite(logp):
#         # print("[DEBUG score_structure] P(Z) is -inf. ABORTING.")
#         return float("-inf"), []

#     logLs = []
#     for i, (root, leaf_to_type) in enumerate(zip(trees, leaf_type_maps)):
#         # print(f"\n--- Processing Tree {i+1}/{len(trees)} ---")
#         B_sets = compute_B_sets(root, leaf_to_type)
#         root_labels = B_sets.get(root, set())
#         # print(f"[DEBUG score_structure] Tree {i+1} has {len(root_labels)} unique types in its leaves.")

#         if not root_labels:
#             # print(f"[DEBUG score_structure] Tree {i+1} has no mapped leaves. LogL = 0.0")
#             logLs.append(0.0)
#             continue
        
#         C_log = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)
#         if not C_log:
#             return float("-inf"), []

#         # Calculate the final log-likelihood directly
#         tree_logL = tree_marginal_from_root_table_log(C_log)
        
#         if not math.isfinite(tree_logL):
#             return float("-inf"), []
        
#         logLs.append(tree_logL + logp)

#     # print(f"=== FINISHED SCORE_STRUCTURE | Total Log Posterior: {sum(logLs)} ===")
#     # print("="*50 + "\n")
#     return sum(logLs), logLs


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

    rng = random.Random(seed)

    # ----- This section is unchanged -----
    if candidate_pool is None:
        candidate_pool = collect_fitch_multis(S, trees, leaf_type_maps)
        if not candidate_pool:
            candidate_pool = [P for P in all_nonempty_subsets(S) if len(P) >= 2]
    pool_set = set(candidate_pool)

    def make_struct(Zset: Set[FrozenSet[str]]) -> "Structure":
        A = _full_edges_for_Z(Zset, unit_drop_edges)
        return Structure(S, Zset, A, unit_drop=unit_drop_edges)

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
    
    current = make_struct(Z0)
    curr_score, _ = score_structure_no_edge_prior(current, trees, leaf_type_maps, priors)
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

        m = rng.choice(block_swap_sizes)
        m = max(1, min(m, len(act), len(ina)))
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
    accepts = 0
    tried = 0
    
    # Helper for pretty printing
    def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

    iterator = range(steps)
    if progress:
        iterator = trange(steps, desc="MCMC (Z-only)", leave=True)

    for it in iterator:
        Zprop = None
        # <<<--- CHANGE 2: Add a variable to store proposal details --->>>
        proposal_details = None

        if priors.potency_mode == "fixed_k" and fixed_k is not None:
            # <<<--- CHANGE 3: Unpack the details from the proposal call --->>>
            Zprop, proposal_details = propose_fixed_k_swap(current.Z_active)
            # Zprop, proposal_details = propose_jaccard_swap(current.Z_active)

            # if proposal_details:
            #     dropped_str = ", ".join(pot_str(p) for p in proposal_details['drop'])
            #     added_str = ", ".join(pot_str(p) for p in proposal_details['add'])
            #     print(f"\n[Proposing Swap]: {dropped_str} -> {added_str}")
        else:
            Zprop = propose_toggle(current.Z_active)

        if Zprop is None:
            tried += 1
            if progress:
                iterator.set_postfix({"logpost": f"{curr_score:.3f}", "acc": f"{(accepts/max(1,tried)):.2f}"})
            continue

        prop_struct = make_struct(Zprop)
        prop_score, _ = score_structure_no_edge_prior(prop_struct, trees, leaf_type_maps, priors)
        
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
            if curr_score > best_score:
                best_struct = current.clone()
                best_score = curr_score

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
        "accept_rate": (accepts / max(1, tried)),
        "inclusion": inclusion,
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

# def _mcmc_worker_Z(args: tuple):
#     """
#     Worker function that creates a cache for a single MCMC chain.
#     """
#     (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
#      steps, burn_in, thin, seed, progress, candidate_pool, block_swap_sizes, fitch_probs) = args

#     # <<< CHANGE 1: CREATE A CACHE FOR THIS WORKER >>>
#     # This cache is local to this single MCMC chain.
#     score_cache = {}

#     return mcmc_map_search(
#         S=S, trees=trees, leaf_type_maps=leaf_type_maps, priors=priors,
#         unit_drop_edges=unit_drop_edges, fixed_k=fixed_k, steps=steps,
#         burn_in=burn_in, thin=thin, seed=seed, progress=True,
#         candidate_pool=candidate_pool, block_swap_sizes=block_swap_sizes,
#         fitch_probs=fitch_probs,
#         score_cache=score_cache  # <<< CHANGE 2: PASS THE CACHE
#     )

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

def run_mcmc_only_A_parallel (
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
            S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
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
            S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
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

    _, per_tree_logs = score_structure(
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
    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
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

# def plot_mcmc_traces(
#     all_stats: List[Dict],
#     burn_in: int,
#     thin: int,
#     title: str,
#     output_path: str
# ):
#     """
#     Plots the log posterior score traces from multiple MCMC chains.

#     Args:
#         all_stats: List of stats dictionaries from each parallel chain.
#         burn_in: The number of burn-in iterations.
#         thin: The thinning interval.
#         title: The title for the plot.
#         output_path: The file path to save the plot image.
#     """
#     fig, ax = plt.subplots(figsize=(12, 7))

#     for i, stats in enumerate(all_stats):
#         scores = stats.get('scores', [])
#         if not scores:
#             continue
        
#         # Calculate the actual iteration numbers for the x-axis
#         iterations = range(burn_in, burn_in + len(scores) * thin, thin)
#         ax.plot(iterations, scores, label=f'Chain {i+1}', alpha=0.8)

#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel("MCMC Iteration", fontsize=12)
#     ax.set_ylabel("Log Posterior Score", fontsize=12)
#     ax.legend()
#     ax.grid(True, linestyle='--', alpha=0.6)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close(fig)
#     print(f"📈 Plot saved to {output_path}")


def process_case(map_idx: int, type_num: int, cells_n: int,
                 priors, iters=100, restarts=5, log_dir: Optional[str]=None,
                 tree_kind: str = "graph", n_jobs: Optional[int] = None):
    # Resolve and validate all inputs (will print what it tries)
    fate_map_path, idx4 = build_fate_map_path(map_idx, type_num, tree_kind=tree_kind)
    tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n, tree_kind=tree_kind)

    # load trees + maps
    trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)

    # run MAP search
    # a compact candidate pool keeps things fast & well-mixed
    print_fitch_potency_probs_once(
        S, trees, leaf_type_maps,
        header=f"\n[Potency ranking] type_{type_num}, map {idx4}, cells_{cells_n}"
    )

    pool = collect_fitch_multis(S, trees, leaf_type_maps)

    # <<<--- CHANGE: Compute Fitch probabilities and convert to a dictionary --->>>
    fitch_probs_list = compute_fitch_potency_probs(S, trees, leaf_type_maps)
    fitch_probs_dict = {p: prob for p, prob in fitch_probs_list}

    # NEW CALL
    # The 'restarts' parameter now controls the number of parallel chains
    bestF_Z, best_score_Z, all_stats_Z = run_mcmc_only_Z_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=False,
        fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
        steps=iters,
        burn_in=(iters*15)//100,
        thin=10,
        base_seed=123,
        candidate_pool=pool,
        block_swap_sizes=(1, 2, 3),
        n_chains= min(os.cpu_count() - 1, restarts),  # Use the 'restarts' argument to set the number of chains
        fitch_probs = fitch_probs_dict
    )

    # if all_stats_Z and log_dir:
    #     plot_mcmc_traces(
    #         all_stats=all_stats_Z,
    #         burn_in=(iters * 15) // 100,
    #         thin=10,
    #         title=f"MCMC Trace for Potency Sets (Z) - Map {idx4}, Cells {cells_n}",
    #         output_path=os.path.join(log_dir, f"trace_Z_type{type_num}_{idx4}_cells{cells_n}.png")
    #     )

    bestF, best_score, all_chain_stats = run_mcmc_only_A_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=False,
        fixed_k=priors.fixed_k if priors.potency_mode == "fixed_k" else None,
        steps=iters,
        burn_in=(iters*15)//100,
        thin=10,
        base_seed=123,
        candidate_pool=pool,
        block_swap_sizes=(1, 2, 3),
        n_chains= min(os.cpu_count() - 1, restarts),  # Use the 'restarts' argument to set the number of chains
        Z = bestF_Z.Z_active
    )

    # if all_chain_stats and log_dir:
    #     plot_mcmc_traces(
    #         all_stats=all_chain_stats,
    #         burn_in=(iters * 15) // 100,
    #         thin=10,
    #         title=f"MCMC Trace for Edges (A) - Map {idx4}, Cells {cells_n}",
    #         output_path=os.path.join(log_dir, f"trace_A_type{type_num}_{idx4}_cells{cells_n}.png")
    #     )

    # For compatibility with your existing result processing,
    # you can get the stats of the best chain if needed,
    # or analyze all_chain_stats for convergence.
    # For now, let's just get the acceptance rate from the first chain as an example.
    stats = all_chain_stats[0] if all_chain_stats else {}

    print("MCMC accept rate:", stats["accept_rate"])
    # posterior inclusion frequency of each candidate potency:
    incl = stats["inclusion"]

    print(f"\n=== BEST MAP for type_{type_num}, map {idx4}, cells_{cells_n} ===")
    multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
                          key=lambda x: (len(x), tuple(sorted(list(x)))))
    print("Active potencies (multi-type):")
    for P in multi_sorted: print("  ", pot_str(P))
    print("Singletons (always active):")
    for t in S: print("  ", "{" + t + "}")

    print("\nEdges:")
    edges = sorted([e for e, v in bestF.A.items() if v == 1],
                   key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
    for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  log posterior: {best_score:.6f}")
    # for i, lg in enumerate(per_tree_logs, 1):
    #     print(f"  Tree {i} log P(T|F*): {lg:.6f}")

    # --- Ground truth scoring ---
    predicted_sets = {p for p in bestF.Z_active if len(p) > 1}

    ground_truth_sets, gt_loss, gt_Z_active, gt_edges= score_given_map_and_trees(
        fate_map_path, trees, meta_paths, fixed_k=priors.fixed_k
    )

    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", ground_truth_sets)

    print("\n=== Ground Truth Directed Edges ===")
    for (u, v) in sorted(gt_edges, key=lambda e: (len(e[0]), tuple(sorted(e[0])), len(e[1]), tuple(sorted(e[1])))):
        print(f"{sorted(list(u))} -> {sorted(list(v))}")

    jd = jaccard_distance(predicted_sets, ground_truth_sets)
    print("\n=== Jaccard Distance ===")
    print(f"Jaccard Distance (Pred vs GT): {jd:.6f}")
    print(f"Predicted map's loss: {best_score:.6f}")
    print(f"Ground truth's loss: {gt_loss:.6f}")

    # --- EDGE METRICS (Predicted vs Ground Truth) ---
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

    # optional logs
    # if log_dir:
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_path = os.path.join(log_dir, f"log_type{type_num}_{idx4}_cells{cells_n}.txt")
    #     with open(log_path, "w") as f:
    #         f.write(f"type_{type_num}, map {idx4}, cells_{cells_n}\n")
    #         f.write(f"Jaccard={jd:.6f}, GT loss={gt_loss:.6f}, Pred loss={best_score:.6f}\n")
    return jd, gt_loss, best_score,edge_jacc,im_s

def main_multi_type(type_nums=[10,14],
                    maps_start=17, maps_end=26,
                    cells_list=[50,100,200],
                    iters = 50,
                    restarts = 4,
                    fixed_k = 5,
                    out_csv="results_types_6_10_14_maps_17_26.csv",
                    log_dir="logs_types",
                    tree_kind: str = "graph"):
    random.seed(7)
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
    results = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type","MapIdx","Cells","Jaccard","GT Loss","Pred Loss","Edge_Jaccard","IM_similarity"])

        for t in type_nums:
            for idx in range(maps_start, maps_end+1):
                for cells in cells_list:
                    try:
                        jd, gt_loss, pred_loss, edge_jacc, im_s = process_case(
                            idx, t, cells, priors,
                            iters=iters, restarts=restarts, log_dir=log_dir,
                            tree_kind=tree_kind, n_jobs= os.cpu_count()-1  # start single-process
                        )
                        writer.writerow([t, idx, cells, f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}", f"{edge_jacc:.6f}", f"{im_s:.6f}"])
                        results.append((t, idx, cells, jd, gt_loss, pred_loss, edge_jacc, im_s))
                    except Exception as e:
                        print(f"[WARN] Failed type_{t} map {idx:04d} cells_{cells}: {repr(e)}")
                        traceback.print_exc()
                        writer.writerow([t, idx, cells, "ERROR","ERROR","ERROR", "ERROR","ERROR"])
                        results.append((t, idx, cells, None,None,None,None,None))


if __name__ == "__main__":
    main_multi_type(
        type_nums=[10],
        maps_start=4,
        maps_end=4,
        cells_list=[50],
        iters = 80,
        restarts = 7,
        fixed_k = 9,
        out_csv="checking.csv",
        log_dir="prac",
        tree_kind="graph"   # or "bin_trees" or "graph"
    )