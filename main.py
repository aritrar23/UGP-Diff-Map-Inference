from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import os
import random
import csv
import json
import math
import itertools
from typing import Iterable,  Dict, Tuple, List, Optional, Set, FrozenSet
from collections import Counter, defaultdict
from tqdm import trange
import traceback
import numpy as np


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

def beta_integral(O:int,D:int)->float:
    # Compute the integral ∫_0^1 p^O (1-p)^D dp, which is the Beta function B(O+1, D+1).
    # We do it in log-space using lgamma for numerical stability:
    #   B(a,b) = Γ(a)Γ(b) / Γ(a+b)
    # Here a = O+1, b = D+1.
    #
    # Used by: tree_marginal_from_root_table() to integrate out Bernoulli p ~ Beta(1,1).
    return math.exp(math.lgamma(O+1)+math.lgamma(D+1)-math.lgamma(O+D+2))

def sparse_convolve_2d(A: Dict[Tuple[int,int],float], B: Dict[Tuple[int,int],float]) -> Dict[Tuple[int,int],float]:
    # Convolution over 2D count tables:
    # Each dict maps (O,D) -> weight. We want:
    #   (A * B)[(o1+o2, d1+d2)] += A[(o1,d1)] * B[(o2,d2)]
    #
    # This is used to combine children's DP messages (sum of counts across subtrees).
    #
    # Edge cases:
    # - If one table is empty, return a copy of the other (identity for convolution).
    if not A: return B.copy()
    if not B: return A.copy()
    out=defaultdict(float)
    for (o1,d1),w1 in A.items():
        for (o2,d2),w2 in B.items():
            out[(o1+o2,d1+d2)] += w1*w2
    return dict(out)

def dp_tree_root_table(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
    prune_eps: float = 0.0
)->Dict[Tuple[int,int],float]:
    # Dynamic program over the tree to build a sparse table at the root:
    #   C[(O,D)] = total weight of all labelings that yield (O,D) at the root,
    # where for a node v labeled by set L, we count:
    #   O_local = |L ∩ B(v)|   (observed hits within v's subtree)
    #   D_local = |L \ B(v)|   (misses: types in L not present under v)
    #
    # The parent-child label constraint is enforced by `Reach`:
    #   If parent has label P, allowed child labels are any L in Reach[P].
    # At the root, P=None means we consider *all* active labels as possible root labels.
    #
    # Pruning:
    # - If prune_eps > 0, we drop entries whose weight < prune_eps * sum(weights) at that node.
    #
    # Used by: score_structure(); its output goes into tree_marginal_from_root_table().

    # Map label -> integer index (for memoization keys).
    label_index={L:i for i,L in enumerate(active_labels)}
    # Memo maps (node-id, parent-label-index-or-(-1 for None)) -> sparse table dict
    memo: Dict[Tuple[int,int], Dict[Tuple[int,int],float]]={}
    def nid(v:TreeNode)->int: return id(v)

    def M(v:TreeNode, P: Optional[FrozenSet[str]])->Dict[Tuple[int,int],float]:
        # Return the sparse (O,D)->weight table for subtree rooted at v,
        # conditioned on: parent of v has label P (P may be None at the root).
        key=(nid(v), -1 if P is None else label_index[P])
        if key in memo: return memo[key]

        if v.is_leaf():
            # For leaves, there is no subtree below: the children "conv" is the identity {(0,0):1}.
            # Note: O_local/D_local are *added* at the PARENT level (where v is processed as a child).
            # So at the leaf node itself, we just return the neutral table.
            memo[key] = {(0,0):1.0}; return memo[key]

        # Bv is the set of observed types that appear anywhere under v (from compute_B_sets()).
        Bv=B_sets[v]
        out=defaultdict(float)

        # Which labels can v take given its parent label P?
        # - If P is None (we're at the root), we try all active labels.
        # - Else we restrict to labels reachable from P according to the potency DAG closure.
        if P is None:
            parent_reach = active_labels
        else:
            parent_reach = list(Reach[P])

        # Try each candidate label L for node v.
        for L in parent_reach:
            # Containment constraint: the observed types in v's subtree must be a subset of L,
            # otherwise L would "claim" types that don't exist under v or miss types that do exist
            # (the model enforces this monotonic consistency).
            if not Bv.issubset(L):  # containment constraint
                continue

            # Local O/D contributions if v is labeled with L:
            #   - Observed hits are types in both L and Bv.
            #   - Misses are types in L that do not appear under v at all.
            o_local=len(L & Bv); d_local=len(L - Bv)

            # Recurse on children conditioned on v's label being L.
            child_tabs=[]
            ok=True
            for u in v.children:
                tab = M(u, L)
                if not tab: ok=False; break
                child_tabs.append(tab)
            if not ok: continue

            # Convolve the children's tables to aggregate counts across subtrees.
            # If there are no children (shouldn't happen for non-leaf), the identity {(0,0):1.0} is used.
            conv = child_tabs[0] if child_tabs else {(0,0):1.0}
            for t in child_tabs[1:]:
                conv = sparse_convolve_2d(conv, t)

            # Add v's local (o_local, d_local) to every child combination.
            for (Oc,Dc),w in conv.items():
                out[(Oc+o_local, Dc+d_local)] += w

        # Optional pruning to keep the table small (drop tiny weights).
        if prune_eps>0 and out:
            total=sum(out.values()); thresh=prune_eps*total
            out={k:v for k,v in out.items() if v>=thresh}

        memo[key]=dict(out); return memo[key]

    # Kick off the DP from the root with P=None (meaning "try all root labels").
    return M(root, None)

def tree_marginal_from_root_table(C: Dict[Tuple[int,int],float])->float:
    # Turn the root's (O,D)->weight table into a scalar probability by integrating out p:
    #   P(T | F) = Σ_{O,D}  weight(O,D) * B(O+1, D+1)
    # where B(·,·) is the Beta function (computed by beta_integral).
    #
    # Used by: score_structure() to compute per-tree likelihoods.
    return sum(w * beta_integral(O,D) for (O,D),w in C.items())

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
    

    
    def propose_swap_potency(self, rng: random.Random,
                         w_fitch: Optional[Dict[FrozenSet[str], float]] = None
                         ) -> Optional["Structure"]:
        """
        Swap one multi-type potency (|P|>=2) out for a new multi-type potency (|P|>=2),
        while preserving the *number* of incident edges removed:
        - count edges removed from the old potency
        - add exactly the same number of edges for the new potency
        - ensure the new potency is not isolated (at least one parent if it's not the root,
            and at least one child)
        Constraints:
        (i) never remove the master/root potency (frozenset(self.S))
        (ii) edges must be admissible (respecting unit_drop)
        """
        ROOT = frozenset(self.S)
        singles = {frozenset([t]) for t in self.S}

        # --- pick what to remove / add ---
        remove_candidates = [P for P in self.Z_active
                            if len(P) >= 2 and P != ROOT]        # (i) don't remove root
        add_candidates = [P for P in self.potencies_multi_all()
                        if len(P) >= 2 and P not in self.Z_active and P != ROOT]

        if not remove_candidates or not add_candidates:
            return None

        P_rm = rng.choice(remove_candidates)
        if w_fitch:
            eps = 1e-12
            weights = [max(w_fitch.get(P, 0.0), eps) for P in add_candidates]
            s = sum(weights)
            # normalize & sample via cumulative
            r = rng.random() * s
            acc = 0.0
            P_add = add_candidates[-1]  # fallback
            for P, w in zip(add_candidates, weights):
                acc += w
                if acc >= r:
                    P_add = P
                    break
        else:
            # fallback: uniform
            P_add = rng.choice(add_candidates)

        # --- clone and remove P_rm; count removed incident edges ---
        new = self.clone()

        # collect incident edges to P_rm BEFORE deletion
        old_in  = [(X, Y) for (X, Y), v in new.A.items() if v == 1 and Y == P_rm]
        old_out = [(X, Y) for (X, Y), v in new.A.items() if v == 1 and X == P_rm]
        n_in, n_out = len(old_in), len(old_out)
        total_removed = n_in + n_out

        # if the node we're removing is completely isolated (rare), we skip this swap
        if total_removed == 0:
            return None

        # actually remove the potency and incident edges
        new.Z_active.remove(P_rm)
        new.A = {e: v for e, v in new.A.items() if P_rm not in e}

        # add the new potency
        new.Z_active.add(P_add)

        # --- compute candidate parents/children for P_add among existing nodes ---
        parents_all  = [X for X in new.Z_active if X != P_add and admissible_edge(X, P_add, new.unit_drop)]
        children_all = [Y for Y in new.Z_active if Y != P_add and admissible_edge(P_add, Y, new.unit_drop)]

        # If there are zero candidates on one side, this swap can't produce a connected node
        if (P_add != ROOT and len(parents_all) == 0) or len(children_all) == 0:
            return None

        # Targets: keep the *total* equal to removed edges.
        # We *try* to match incoming/outgoing counts but can rebalance to satisfy connectivity.
        target_in, target_out = n_in, n_out

        # Ensure connectivity: at least one incoming (unless P_add is root), and one outgoing
        if P_add != ROOT and target_in == 0:
            if target_out > 0:
                target_in, target_out = 1, target_out - 1
            else:
                # can't add any edge while keeping the same total
                return None
        if target_out == 0:
            if target_in > (1 if P_add != ROOT else 0):
                target_out, target_in = 1, target_in - 1
            else:
                return None

        # Cap by availability
        target_in  = min(target_in,  len(parents_all))
        target_out = min(target_out, len(children_all))

        # If even using everything we can't reach the required total, give up
        if target_in + target_out < total_removed:
            # try to fill the remainder by relaxing split, but not exceeding candidates
            spare_parents  = len(parents_all)  - target_in
            spare_children = len(children_all) - target_out
            need = total_removed - (target_in + target_out)
            take_p = min(need, spare_parents);  need -= take_p;  target_in  += take_p
            take_c = min(need, spare_children); need -= take_c;  target_out += take_c
            if need > 0:
                return None

        # Now sample distinct parents/children to meet targets
        rng.shuffle(parents_all)
        rng.shuffle(children_all)
        chosen_parents  = parents_all[:target_in]
        chosen_children = children_all[:target_out]

        # Wire them
        for X in chosen_parents:
            new.A[(X, P_add)] = 1
        for Y in chosen_children:
            new.A[(P_add, Y)] = 1

        # Sanity: exact count
        # (We added exactly target_in + target_out == total_removed edges)
        assert sum(1 for e, v in new.A.items() if v == 1 and (e[0] == P_add or e[1] == P_add)) == total_removed

        # --- ensure the node isn't isolated in the global sense (root -> ... -> singleton via P_add) ---
        # quick check: after these edges, recompute reachability and verify a path exists
        new.recompute_reach()
        # path from ROOT to P_add (unless P_add is ROOT)
        root_ok = (P_add == ROOT) or (P_add in new.Reach[ROOT])
        # from P_add to at least one singleton
        child_to_leaf_ok = any((frozenset([t]) in new.Reach[P_add]) for t in self.S)

        if not (root_ok and child_to_leaf_ok):
            # Try one lightweight adjustment within the same edge budget:
            # - If root_ok is false and we have an admissible parent reachable from ROOT that we didn't use, swap one edge.
            if not root_ok and P_add != ROOT:
                candidates = [X for X in parents_all if X in new.Reach[ROOT] and (X, P_add) not in new.A]
                if candidates:
                    # replace a parent edge if we already used some, otherwise replace a child edge
                    repl_src = None
                    for X, Y in list(new.A.keys()):
                        if new.A[(X, Y)] == 1 and Y == P_add:
                            repl_src = (X, Y); break
                    if repl_src is None:
                        for X, Y in list(new.A.keys()):
                            if new.A[(X, Y)] == 1 and X == P_add:
                                repl_src = (X, Y); break
                    if repl_src is not None:
                        del new.A[repl_src]
                        new.A[(rng.choice(candidates), P_add)] = 1

            # - If leaf path is false, try to ensure we have at least one child that can reach a singleton
            new.recompute_reach()
            if not any((frozenset([t]) in new.Reach[P_add]) for t in self.S):
                cand_children = []
                for Y in children_all:
                    # check if Y or something below Y is a singleton
                    if any((frozenset([t]) in new.Reach[Y]) for t in self.S):
                        cand_children.append(Y)
                cand_children = [Y for Y in cand_children if (P_add, Y) not in new.A]
                if cand_children:
                    # replace one child edge with a better child
                    repl = None
                    for X, Y in list(new.A.keys()):
                        if new.A[(X, Y)] == 1 and X == P_add:
                            repl = (X, Y); break
                    if repl is None:
                        # as last resort, replace a parent edge
                        for X, Y in list(new.A.keys()):
                            if new.A[(X, Y)] == 1 and Y == P_add:
                                repl = (X, Y); break
                    if repl is not None:
                        del new.A[repl]
                        new.A[(P_add, rng.choice(cand_children))] = 1
                        new.recompute_reach()

            # final check
            root_ok = (P_add == ROOT) or (P_add in new.Reach[ROOT])
            child_to_leaf_ok = any((frozenset([t]) in new.Reach[P_add]) for t in self.S)
            if not (root_ok and child_to_leaf_ok):
                # unable to satisfy connectivity without changing edge count -> reject
                return None

        return new


    # def propose_swap_potency(self, rng:random.Random)->Optional["Structure"]:
    #     # Propose: swap out one existing multi-type potency for a different one not currently active.
    #     # Useful in fixed-k mode to keep the number of multi-type nodes constant while exploring.
    #     remove_candidates = [P for P in self.Z_active if len(P)>=2]
    #     add_candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
    #     if not remove_candidates or not add_candidates: return None
    #     P_rm = rng.choice(remove_candidates)
    #     P_add = rng.choice(add_candidates)
    #     new = self.clone()
    #     new.Z_active.remove(P_rm)
    #     new.A = {e:v for e,v in new.A.items() if P_rm not in e}
    #     new.Z_active.add(P_add)
    #     new.recompute_reach()
    #     return new

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
    
    def propose_add_edge_guided(self, rng: random.Random,
                            w: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float]) -> Optional["Structure"]:
        # candidates missing edges
        pairs = [e for e in self.all_edge_pairs() if self.A.get(e,0)==0]
        if not pairs: return None
        # weights from aggregated transitions if available; small epsilon fallback
        eps = 1e-6
        ws = []
        for (P,Q) in pairs:
            ws.append(max(w.get((P,Q), 0.0), eps))
        # sample proportional to weight
        total = sum(ws)
        r = rng.random() * total
        acc = 0.0
        for (e, we) in zip(pairs, ws):
            acc += we
            if acc >= r:
                new = self.clone()
                new.A[e] = 1
                new.recompute_reach()
                return new
        # fallback
        e = rng.choice(pairs)
        new = self.clone(); new.A[e]=1; new.recompute_reach(); return new


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
    #
    # Posterior:
    #   log P(F | data) ∝ log P(F) + Σ_T log P(T | F)
    # where:
    #   - log P(F) splits into:
    #       * log P(Z_active): prior over which potencies (nodes) are active
    #       * log P(A | Z_active): prior over which admissible edges exist
    #   - log P(T | F) is the tree likelihood computed by the DP + Beta integral.
    #
    # Inputs
    #   struct         : Structure holding S, Z_active, A, labels_list (sorted Z), and Reach (transitive closure)
    #   trees          : list of tree roots (each a TreeNode) to be scored under F
    #   leaf_type_maps : parallel list of dicts mapping leaf name -> observed type (as strings)
    #   priors         : Priors object encapsulating hyperparameters and prior computations
    #   prune_eps      : optional DP pruning threshold (passed down to dp_tree_root_table)
    #
    # Outputs
    #   (log_post, per_tree_logLs)
    #     log_post      : total log posterior = log prior + sum of per-tree log likelihoods
    #     per_tree_logLs: list of per-tree log-likelihoods log P(T | F) (one float per tree)
    #
    # Where this is used:
    #   - Called repeatedly from map_search() to evaluate proposals during the
    #     hill-climb / simulated annealing over (Z_active, A).

    # ---- Prior over structure F ----
    # log P(Z_active): fixed-k or Bernoulli over multi-type potencies (singles implicit/free)
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    # If Z violates the prior's support (e.g., wrong k in fixed-k), we get -inf; bail early.
    if not math.isfinite(logp): return float("-inf"), []
    # log P(A | Z_active): independent Bernoulli(rho) over admissible edges under unit_drop flag
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    # ---- Likelihood over all trees ----
    logLs=[]
    for root, leaf_to_type in zip(trees, leaf_type_maps):
        # Precompute B_sets[v] = set of observed types anywhere under node v.
        # Robust behavior: leaves missing from the map contribute empty sets (ignored).
        B_sets = compute_B_sets(root, leaf_to_type)

        # Neutral-evidence shortcut:
        # If the root accumulates *no observed types at all* (e.g., map has no usable labels
        # for this tree after filtering), then this tree carries no information about F.
        # We treat it as contributing log-likelihood 0.0 (i.e., multiplicative factor 1).
        # This prevents crashes where P(T|F) would be numerically zero for all F.
        root_labels = B_sets.get(root, set())
        if not root_labels:
            logLs.append(0.0)
            continue

        # Dynamic program over labelings constrained by struct.Reach:
        # Builds a sparse table at the root: C[(O,D)] = total weight for that count pair,
        # where O = observed hits, D = misses, given the root label and subtree constraints.
        C = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)

        # Integrate out p ~ Beta(1,1): P(T | F) = Σ_{(O,D)} C[(O,D)] * B(O+1, D+1).
        P_T = tree_marginal_from_root_table(C)

        # If numerical underflow or structural inconsistency yields P_T <= 0, the score is invalid.
        if P_T <= 0 or not math.isfinite(P_T):
            return float("-inf"), []

        # Accumulate per-tree log-likelihood
        logLs.append(math.log(P_T))

    # Total posterior score = log prior + sum of per-tree log-likelihoods
    return logp + sum(logLs), logLs



# ----------------------------
# Annealed stochastic search
# ----------------------------

def fitch_potency_probs_dict(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
) -> Dict[FrozenSet[str], float]:
    """
    Compute per-potency probabilities from union-Fitch labeling, using the same
    row_sum logic as init_progenitors_union_fitch (parent transition mass).
    Returns a dict {potency_set -> probability} that sums to 1 (over keys with >0 mass).
    Potencies never seen as sources get 0.0.
    """
    row_sum: Dict[frozenset, float] = defaultdict(float)

    for tree, ltm in zip(trees, leaf_type_maps):
        assign_union_potency(tree, ltm)        # sets v.potency on nodes
        C_T = per_tree_transition_counts(tree) # counts parent->child changes only
        T = sum(C_T.values())
        if T == 0:
            continue
        for (i_set, _j_set), cnt in C_T.items():
            row_sum[i_set] += cnt / T

    total = sum(row_sum.values())
    if total <= 0:
        return {}  # no signal; caller should handle smoothing

    return {P: row_sum[P] / total for P in row_sum}

    
def map_search(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    priors: Priors,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    init_seed: int = 0,
    iters: int = 500,
    restarts: int = 3,
    temp_init: float = 1.0,
    temp_decay: float = 0.995,
    move_probs = (0.25, 0.25, 0.25, 0.25),
    prune_eps: float = 0.0,
    progress: bool = True,
    plot_dir: Optional[str] = None,   # <--- NEW
    run_tag: Optional[str] = None,    # <--- NEW (helps name files)
    guided_edge_prob: float = 0.7
    # stagnation_patience: int = 30,
    # reheat_factor: float = 2.0,
    # max_reheats: int = 5,
    # kick_edges: int = 5,       # how many random edge toggles in a kick
    # kick_swaps: int = 2       # how many potency swaps in a kick
):
    rng = random.Random(init_seed)

    best_global = None
    best_score = float("-inf")
    best_logs = None

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    for rs in range(restarts):
        # --- init structure
        # print(rs)
        

        if priors.potency_mode=="fixed_k":
            agg_w, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
        else:
            agg_w, base = ({},build_Z_active(S, fixed_k=0, max_potency_size=len(S), seed=rng.randint(0,10**9)))
            Z = base
        A = build_mid_sized_connected_dag(Z,keep_prob = 0.3,rng = None)
        current = Structure(S, Z, A, unit_drop=unit_drop_edges)
        curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)

        

        # fallback: if invalid, keep sampling until valid
        attempts = 0
        while not math.isfinite(curr_score) and attempts < 720:
            print(f"This is attempt number: {attempts}")
            aggregated_transitions, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
            A = build_mid_sized_connected_dag(Z,keep_prob = 0.3,rng = None)
            # A = {}
            # print(f"Z:{Z}") 
            # print(f"A:{A}")
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)
            # print(curr_score)
            attempts += 1

        if not math.isfinite(curr_score):
            raise RuntimeError("Failed to initialize a valid structure; consider easing settings.")
        
        # no_improve = 0
        # reheats_used = 0
        local_best_score = curr_score

        local_best = current.clone()
        
        if (best_score < curr_score):
            best_score = curr_score
            best_global = current.clone()

        tau = temp_init
        addP, rmP, addE, rmE = move_probs

        # ---- history containers for this restart ----
        curr_hist: List[float] = [curr_score]   # score of the current state each iteration
        best_hist: List[float] = [best_score]   # running best score (global or local; choose one)

        # iterator respects the 'progress' flag
        iterator = trange(iters, desc=f"Restart {rs+1}/{restarts}", leave=True) if progress else range(iters)

        for it in iterator:
            # choose move
            prop = None
            r = rng.random()

            
            swapping = False

            if priors.potency_mode=="fixed_k":
                agg_w, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
                # print("\n[Init] Potencies from Fitch:")
                # for P in sorted(Z, key=lambda x: (len(x), tuple(sorted(x)))):
                #     print(" ", P)

                if r < addE:
                    prop = current.propose_add_edge(rng)
                elif r < addE + rmE:
                    prop = current.propose_remove_edge(rng)
                else:
                    # print("Trying to swap potency")
                    swapping = True
                    prop = current.propose_swap_potency(rng)

                    if prop is None:
                        hello = 1
                        # print("[SWAP] No valid swap proposal (constraints prevented construction).")
                    else:
                        def _pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"
                        multis_prop = sorted(
                            [P for P in prop.Z_active if len(P) >= 2],
                            key=lambda x: (len(x), tuple(sorted(list(x))))
                        )
                        # print("[SWAP] Proposed Z' (multi-type potencies):")
                        # for P in multis_prop:
                        #     print("   ", _pot_str(P))
            else:
                if r < addP:
                    prop = current.propose_add_potency(rng)
                elif r < addP + rmP:
                    prop = current.propose_remove_potency(rng)
                elif r < addP + rmP + addE:
                    prop = current.propose_add_edge(rng)
                else:
                    prop = current.propose_remove_edge(rng)

            if prop is None:
                tau *= temp_decay
                if progress:
                    iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})
                continue

            prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors, prune_eps)

            delta = prop_score - curr_score
            # if (swapping == True):
            #     print(f"Current score: {curr_score}")
            #     print(f"Prop score: {prop_score}")
            accept = (delta >= 0) or (rng.random() < math.exp(delta / max(tau,1e-12)))
            if accept:
                current = prop
                curr_score = prop_score
                
                # if (swapping == True):
                #     print("Yay! Swapped successfully!")

                if curr_score > local_best_score:
                    local_best = current.clone()
                    local_best_score = curr_score

                if curr_score > best_score:
                    best_global = current.clone()
                    best_score = curr_score
                    best_logs = None  # compute later if needed

            tau *= temp_decay

            
            # print(f"rs:{rs},curr{curr_score}")
            # after deciding accept/reject and updating curr_score / best_score:
            curr_hist.append(curr_score)
            best_hist.append(best_score)   # or local_best_score, if you prefer per-restart best

            if progress:
                iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})

        if plot_dir is not None:
            plt.figure(figsize=(6.5, 4.0))
            plt.plot(curr_hist, label="current score")
            plt.plot(best_hist, label="best score")
            plt.xlabel("Iteration")
            plt.ylabel("Log posterior")
            plt.title(f"Score vs Iteration — restart {rs+1}" + (f" ({run_tag})" if run_tag else ""))
            plt.legend()
            fname = f"score_vs_iter_restart_{rs+1}" + (f"_{run_tag}" if run_tag else "") + ".png"
            out_path = os.path.join(plot_dir, fname)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
        
    # after restarts, recompute detailed logs for best_global

    final_score, logLs = score_structure(best_global, trees, leaf_type_maps, priors, prune_eps)
    return best_global, final_score, logLs


def _map_search_worker(args):
    (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
     init_seed, iters, restarts, temp_init, temp_decay, move_probs, prune_eps,
     plot_dir, run_tag) = args  # <--- added

    return map_search(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=unit_drop_edges,
        fixed_k=fixed_k,
        init_seed=init_seed,
        iters=iters,
        restarts=restarts,
        temp_init=temp_init,
        temp_decay=temp_decay,
        move_probs=move_probs,
        prune_eps=prune_eps,
        progress=True,
        plot_dir=plot_dir,    # <--- added
        run_tag=run_tag,      # <--- added
    )


def map_search_parallel(
    S: List[TreeNode],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
    priors: Priors,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    init_seed: int = 0,
    iters: int = 500,
    restarts: int = 12,
    temp_init: float = 1.0,
    temp_decay: float = 0.995,
    move_probs = (0.25, 0.25, 0.25, 0.25),
    prune_eps: float = 0.0,
    n_jobs: Optional[int] = None,
    plot_root: Optional[str] = None,     # <--- NEW
    run_tag: Optional[str] = None,       # <--- NEW
):

    """
    Parallelizes restarts across processes and returns the best result.
    NOTE: On Windows/macOS, call this under `if __name__ == "__main__":` to avoid spawn issues.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    # Split restarts across jobs
    per_job = [restarts // n_jobs] * n_jobs
    for i in range(restarts % n_jobs):
        per_job[i] += 1
    per_job = [r for r in per_job if r > 0]
    n_jobs = len(per_job)

    # Unique seeds per worker to diversify trajectories
    seeds = [init_seed + 10_000 * i for i in range(n_jobs)]

    tasks = []
        
    for r, seed in zip(per_job, seeds):
        worker_plot_dir = None
        worker_tag = run_tag
        if plot_root is not None:
            worker_plot_dir = os.path.join(plot_root, f"seed_{seed}")
            os.makedirs(worker_plot_dir, exist_ok=True)
            # augment the tag with the seed so filenames are unique and traceable
            worker_tag = (run_tag + f"_seed{seed}") if run_tag else f"seed{seed}"

        tasks.append((S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
                      seed, iters, r, temp_init, temp_decay, move_probs, prune_eps,
                      worker_plot_dir, worker_tag))  # <--- pass through

    best_global = None
    best_score = float("-inf")
    best_logs = None

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_map_search_worker, t) for t in tasks]
        for fut in as_completed(futures):
            bestF, score, logs = fut.result()
            if score > best_score:
                best_global, best_score, best_logs = bestF, score, logs

    return best_global, best_score, best_logs

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
    import os, csv, json

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
    
# ----------------------------
# Demo: random trees + search
# ----------------------------


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

    # Print potency definitions
    # print("\n=== Potency definitions (expanded) ===")
    # for pid in sorted(potency_def, key=lambda x: (len(x), x)):
    #     s = ",".join(sorted(potency_def[pid]))
    #     print(f"  {pid} := {{{s}}}")

    # Load trees and leaf maps
    # trees = [
    #     read_newick_file("./0002_tree_0.txt"),
    #     read_newick_file("./0002_tree_1.txt"),
    #     read_newick_file("./0002_tree_2.txt"),
    #     read_newick_file("./0002_tree_3.txt"),
    #     read_newick_file("./0002_tree_4.txt")
    # ]
    # meta_paths = [
    #     "./0002_meta_0.txt",
    #     "./0002_meta_1.txt",
    #     "./0002_meta_2.txt",
    #     "./0002_meta_3.txt",
    #     "./0002_meta_4.txt"
    # ]
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

        # Ground-truth node & edge sets (over potencies, including singletons)
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

# --- CASE RUNNER ---

# def process_case(map_idx: int, type_num: int, cells_n: int,
#                  priors, iters=100, restarts=5, log_dir: Optional[str]=None):

#     fate_map_path, idx4 = build_fate_map_path(map_idx, type_num)
#     tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n)

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



def process_case(map_idx: int, type_num: int, cells_n: int,
                 priors, iters=100, restarts=5, log_dir: Optional[str]=None,
                 tree_kind: str = "graph", n_jobs: Optional[int] = None,
                 baseline_only: bool = False):  # <--- NEW
                 
    # Resolve and validate all inputs (will print what it tries)
    fate_map_path, idx4 = build_fate_map_path(map_idx, type_num, tree_kind=tree_kind)
    tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n, tree_kind=tree_kind)

    # load trees + maps
    trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)

        # load trees + maps
    # trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)

    # --- PRINT FITCH POTENCY PROBABILITIES ONCE (per cell-diff map) ---
    print_fitch_potency_probs_once(
        S, trees, leaf_type_maps,
        header=f"\n[Potency ranking] type_{type_num}, map {idx4}, cells_{cells_n}"
    )


    # default: single-process on Windows for easier debugging (you can bump later)
    # if n_jobs is None:
    #     n_jobs = 1

    # run MAP search
        # Build a descriptive plot root (one folder per case)


        # --- BASELINE (Fitch-like init only; no iterations, no restarts) ---
    if baseline_only:
        # Fitch union labeling + top-(k-1) progenitors
        agg_w, Z_init = init_progenitors_union_fitch(S, trees, leaf_type_maps, priors.fixed_k)

        # Predicted sets = multi-type potencies from the Fitch init (exclude singletons)
        predicted_sets = {P for P in Z_init if len(P) >= 2}

        # Ground truth sets and GT loss (uses the provided fate map + data)
        ground_truth_sets, gt_loss = score_given_map_and_trees(
            fate_map_path, trees, meta_paths, fixed_k=priors.fixed_k
        )

        # Jaccard (baseline)
        jd = jaccard_distance(predicted_sets, ground_truth_sets)

        print("\n=== BASELINE (Fitch init only) ===")
        pretty_print_sets("Predicted Sets (Fitch init)", predicted_sets)
        pretty_print_sets("Ground Truth Sets", ground_truth_sets)
        print(f"\nJaccard Distance (Baseline): {jd:.6f}")
        print(f"Ground truth loss: {gt_loss:.6f}")

        # Optionally log
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"baseline_type{type_num}_{idx4}_cells{cells_n}.txt")
            with open(log_path, "w") as f:
                f.write(f"[BASELINE] type_{type_num}, map {idx4}, cells_{cells_n}\n")
                f.write(f"Jaccard={jd:.6f}, GT loss={gt_loss:.6f}\n")

        # Return a triple consistent with the search path (jd, gt_loss, pred_loss)
        # Here pred_loss is None (we're not scoring a structure in baseline).
        return jd, gt_loss, 0

    case_plot_root = os.path.join(
        "plots",
        f"type_{type_num}",
        f"map_{idx4}",
        f"cells_{cells_n}"
    )
    os.makedirs(case_plot_root, exist_ok=True)

    bestF, best_score, per_tree_logs = map_search_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=False,
        fixed_k=priors.fixed_k if priors.potency_mode=="fixed_k" else None,
        init_seed=123,
        iters=iters,
        restarts=restarts,
        temp_init=1.0,
        temp_decay=0.995,
        move_probs=(0.3, 0.2, 0.3, 0.2),
        prune_eps=0.0,
        n_jobs=n_jobs,
        plot_root=case_plot_root,                           # <--- NEW
        run_tag=f"type{type_num}_map{idx4}_cells{cells_n}"  # <--- NEW
    )

    # --- Pretty print inferred map ---
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
    for i, lg in enumerate(per_tree_logs, 1):
        print(f"  Tree {i} log P(T|F*): {lg:.6f}")

    # --- Ground truth scoring ---
    predicted_sets = {p for p in bestF.Z_active if len(p) > 1}


    ground_truth_sets, gt_loss, gt_Z_active, gt_edges = score_given_map_and_trees(
        fate_map_path, trees, meta_paths, fixed_k=priors.fixed_k
    )


    # ground_truth_sets, gt_loss = score_given_map_and_trees(
    #     fate_map_path, trees, meta_paths, fixed_k=priors.fixed_k
    # )

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
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"log_type{type_num}_{idx4}_cells{cells_n}.txt")
        with open(log_path, "w") as f:
            f.write(f"type_{type_num}, map {idx4}, cells_{cells_n}\n")
            f.write(f"Jaccard={jd:.6f}, GT loss={gt_loss:.6f}, Pred loss={best_score:.6f}\n")
    return jd, gt_loss, best_score, edge_jacc, im_s


def main_multi_type(type_nums=[6,10,14],
                    maps_start=17, maps_end=26,
                    cells_list=[50,100,200],
                    iters = 100,
                    restarts = 7,
                    out_csv="results_types_6_10_14_maps_17_26.csv",
                    log_dir="logs_types",
                    tree_kind: str = "graph",
                    fixed_k: int = 7,
                    baseline_only: bool = False ):    # <--- NEW


    random.seed(7)
    priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)
    results = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type","MapIdx","Cells","Jaccard","GT Loss","Pred Loss","Edge Jaccard","IM Similarity"])

        for t in type_nums:
            for idx in range(maps_start, maps_end+1):
                for cells in cells_list:
                    try:
                        jd, gt_loss, pred_loss, edge_jacc, im_s = process_case(
                            idx, t, cells, priors,
                            iters=iters, restarts=restarts, log_dir=log_dir,
                            tree_kind=tree_kind, 
                            n_jobs= os.cpu_count()-1,
                            baseline_only=baseline_only  # start single-process
                            # n_jobs= 1  # start single-process
                        )
                        writer.writerow([t, idx, cells, f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}",
                        f"{edge_jacc:.6f}", f"{im_s:.6f}"])

                        results.append((t, idx, cells, jd, gt_loss, pred_loss))
                    except Exception as e:
                        print(f"[WARN] Failed type_{t} map {idx:04d} cells_{cells}: {repr(e)}")
                        traceback.print_exc()
                        writer.writerow([t, idx, cells, "ERROR","ERROR","ERROR"])
                        results.append((t, idx, cells, None,None,None))

    print(f"\nSummary saved to {out_csv}")
    print("\n================= Summary Table =================")
    print(f"{'Type':<6} {'Map':<6} {'Cells':<7} {'Jaccard':<12} {'GT Loss':<14} {'Pred Loss':<14}")
    for t, idx, cells, jd, gt, pr in results:
        if jd is None:
            print(f"{t:<6} {idx:<6} {cells:<7} {'ERROR':<12} {'ERROR':<14} {'ERROR':<14}")
        else:
            print(f"{t:<6} {idx:<6} {cells:<7} {jd:<12.6f} {gt:<14.6f} {pr:<14.6f}")

if __name__ == "__main__":
    main_multi_type(
        type_nums=[10],
        maps_start=2,
        maps_end = 7,
        cells_list=[50],
        iters = 50,
        restarts = 7,
        out_csv="10_50_2_7_all_loss.csv",
        # log_dir="logs_types",
        tree_kind="graph",   # or "bin_trees" or "graph"
        fixed_k=9,
        # baseline_only = True   # <--- choose k here
    )
