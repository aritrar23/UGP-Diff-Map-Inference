#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EM-based Carta inference with exact marginalization over labelings.
Same input handling as original script (main_multi/process_case style).
"""

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
# -------------------
# Tree utilities
# -------------------

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


class Priors:
    def __init__(self, potency_mode="fixed_k", fixed_k=2, pi_P=0.25, rho=0.25):
        self.potency_mode=potency_mode
        self.fixed_k=fixed_k
        self.pi_P=pi_P
        self.rho=rho

    def log_prior_Z(self, S, Z_active):
        singles = {frozenset([t]) for t in S}
        multis = [P for P in Z_active if len(P) >= 2]
        all_multis = [P for P in all_nonempty_subsets(S) if len(P) >= 2]

        if self.potency_mode=="fixed_k":
            k=len(multis)
            if k!=self.fixed_k:
                return float("-inf")
            total = math.comb(len(all_multis), k)
            return -math.log(total) if total>0 else float("-inf")
        else:
            k_log=0.0
            for P in all_multis:
                if P in Z_active: k_log += math.log(self.pi_P)
                else: k_log += math.log(1-self.pi_P)
            return k_log

    def log_prior_A(self, Z_active, A, unit_drop=True):
        labels=list(Z_active)
        logp=0.0
        for P in labels:
            for Q in labels:
                if admissible_edge(P,Q,unit_drop):
                    a = 1 if A.get((P,Q),0)==1 else 0
                    logp += math.log(self.rho) if a==1 else math.log(1-self.rho)
        return logp


class TreeNode:
    def __init__(self,name=None):
        self.name=name; self.children=[]; self.parent=None
    def add_child(self,c): self.children.append(c); c.parent=self
    def is_leaf(self): return len(self.children)==0


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
    """
    Drop a synthetic 'root' node (if present) from adjacency for building F.
    Keeps everything as strings.
    """
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if not isinstance(ch, list) or len(ch) != 1:
            raise ValueError("Synthetic 'root' must have exactly one child")
        del adj2["root"]
    return adj2

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

def read_newick_file(path:str)->TreeNode:
    with open(path,"r") as f: return parse_newick(f.read().strip())

def collect_leaf_names(root): 
    out=[]
    def dfs(v): 
        if v.is_leaf(): out.append(v.name)
        else: [dfs(c) for c in v.children]
    dfs(root); return out

def compute_B_sets(root, leaf_map: Dict[str,str]) -> Dict[TreeNode, Set[str]]:
    B={}
    def post(v):
        if v.is_leaf():
            t=leaf_map.get(v.name); B[v]={t} if t else set(); return B[v]
        acc=set(); [acc.update(post(c)) for c in v.children]; B[v]=acc; return B[v]
    post(root); return B


def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set = None) -> frozenset:
    """
    Resolve a potency id to the frozenset of base (negative-string) types.

    Rules:
      - If id starts with "-", it's a base type: returns {id}.
      - Otherwise it must appear in comp_map:
          * if comp_map[id] is a list  -> union-resolve all children
          * else (single)              -> resolve that single child
      - Detects cycles via 'visiting'.
      - Uses 'memo' for caching results.
    """
    if visiting is None:
        visiting = set()

    id_str = str(id_str)

    # base type
    if id_str.startswith("-"):
        return frozenset([id_str])

    # memoized
    if id_str in memo:
        return memo[id_str]

    # cycle detection / existence
    if id_str in visiting:
        raise ValueError(f"Cycle detected while resolving potency '{id_str}'")
    if id_str not in comp_map:
        raise ValueError(f"Missing comp {id_str}")

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
    Build F = (Z_active, A) from:
      - adj: adjacency dictionary using ids (strings), possibly including a synthetic 'root'
      - comp_map: composition map id -> (list of ids | single id | base '-N')
    Returns:
      Z_active: set of potency frozensets (includes all singletons + multis from comp_map)
      A      : adjacency over those frozensets (only admissible edges kept)
      base_types (sorted list of base ids like '-1', '-5', ...)
      potency_id_to_set: dict mapping each positive id to its expanded frozenset
    """
    # 1) Drop synthetic root if present
    adj = _normalize_adj_remove_synthetic_root(adj)

    # 2) Collect all ids we need to resolve (from comp_map and adj)
    ids_seen = set(map(str, comp_map.keys()))
    for u, chs in adj.items():
        ids_seen.add(str(u))
        if isinstance(chs, list):
            for v in chs:
                ids_seen.add(str(v))
        else:
            ids_seen.add(str(chs))

    # 3) Resolve ids to base-type frozensets
    memo = {}
    potency_id_to_set = {}
    base_types = set()

    for idv in ids_seen:
        idv = str(idv)
        s = _resolve_id_to_set(idv, comp_map, memo)  # <-- no 'visiting' arg passed
        # store mapping for positive ids
        if not idv.startswith("-"):
            potency_id_to_set[idv] = s
        # collect base types
        for t in s:
            if t.startswith("-"):
                base_types.add(t)

    # 4) Build Z_active:
    #    - all singletons for each base type
    #    - all multi-type potencies (size >= 2) that appear via comp_map expansion
    Z_active = {frozenset([t]) for t in base_types}
    for pid, s in potency_id_to_set.items():
        if len(s) >= 2:
            Z_active.add(s)

    # 5) Build A over expanded sets, only keeping admissible edges
    A = {}

    def _id_to_set(x: str) -> frozenset:
        x = str(x)
        if x.startswith("-"):
            return frozenset([x])
        # unknown id (not in comp_map) shouldn't happen if we resolved above
        if x not in potency_id_to_set:
            # try to resolve on the fly (defensive)
            potency_id_to_set[x] = _resolve_id_to_set(x, comp_map, memo)
        return potency_id_to_set[x]

    for u, chs in adj.items():
        Pu = _id_to_set(u)
        if isinstance(chs, list):
            for v in chs:
                Qv = _id_to_set(v)
                if admissible_edge(Pu, Qv, unit_drop_edges):
                    A[(Pu, Qv)] = 1
        else:
            # in case adj used single child instead of a list
            Qv = _id_to_set(chs)
            if admissible_edge(Pu, Qv, unit_drop_edges):
                A[(Pu, Qv)] = 1

    return Z_active, A, sorted(base_types), potency_id_to_set

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

def score_structure(struct, trees, leaf_type_maps, priors, prune_eps=0.0):
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp):
        # print("[DEBUG] score_structure: prior over Z is -inf (violates prior support)")
        return float("-inf"), []
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    logLs = []
    # print("\n[DEBUG] score_structure — per-tree details")
    for t_idx, (root, leaf_to_type) in enumerate(zip(trees, leaf_type_maps)):
        # basic alignment
        leaves = collect_leaf_names(root)
        mapped = {l for l in leaves if l in leaf_to_type}
        missing = [l for l in leaves if l not in leaf_to_type][:5]
        # print(f"  Tree {t_idx}: leaves={len(leaves)}, mapped={len(mapped)}, missing_sample={missing}")

        # DP inputs
        B_sets = compute_B_sets(root, leaf_to_type)
        root_labels = B_sets.get(root, set())
        # print(f"    B_sets[root] size = {len(root_labels)}; B_sets[root] = {sorted(root_labels)}")

        if not root_labels:
            # This is the exact branch that yields a 0.0 log-likelihood contribution
            print("    [INFO] No observed types under root → contributes 0.0 to total loss.")
            logLs.append(0.0)
            continue

        C = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)
        P_T = tree_marginal_from_root_table(C)
        if P_T <= 0 or not math.isfinite(P_T):
            print("    [WARN] P(T|F) non-positive or non-finite → returning -inf")
            return float("-inf"), []
        lg = math.log(P_T)
        logLs.append(lg)
        # print(f"    log P(T|F) = {lg:.6f}")

    total = logp + sum(logLs)
    # print(f"[DEBUG] score_structure — sum per-tree logL = {sum(logLs):.6f}, log prior pieces included.")
    return total, logLs

def score_given_map_and_trees(txt_path: str, trees, meta_paths, fixed_k,
                              unit_drop_edges=False):
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

    # 3) root id (not actually used below, but parsed for completeness)
    root_id = objs[3]
    if isinstance(root_id, dict) and "root_id" in root_id:
        root_id = root_id["root_id"]
    root_id = str(root_id)

    # Build Z, A, and potency definitions
    Z_from_map, A_from_map, base_types_map, potency_def = _build_ZA_from_txt(
        adj=adj,
        comp_map=comp_map,
        unit_drop_edges=unit_drop_edges
    )

    # Load trees and leaf maps
    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]
    base_types_data = sorted({str(t) for m in leaf_type_maps for t in m.values()})

    # --- DEBUG: universe sanity ---
    # print("\n[DEBUG] score_given_map_and_trees — potency/type universes")
    # print(f"  base_types_map  (from GT map): {base_types_map}")
    # print(f"  base_types_data (from meta):   {base_types_data}")
    # only_in_map  = sorted(set(base_types_map)  - set(base_types_data))
    # only_in_data = sorted(set(base_types_data) - set(base_types_map))
    # if only_in_map:
    #     print(f"  [WARN] types only in GT map, not in meta: {only_in_map}")
    # if only_in_data:
    #     print(f"  [WARN] types only in meta, not in GT map: {only_in_data}")

    # Merge sets for structure
    S_all = sorted(set(base_types_map) | set(base_types_data))
    Z_active = set(Z_from_map) | {frozenset([t]) for t in S_all}
    A = dict(A_from_map)

    struct = Structure(S=S_all, Z_active=Z_active, A=A, unit_drop=unit_drop_edges)
    dummy_priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)

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

    potency_sets = {frozenset(members) for members in potency_def.values()}
    return potency_sets, total_ll

# -------------------
# Potency structure
# -------------------
def admissible_edge(P,Q,unit_drop=True):
    return (Q!=P and Q.issubset(P) and len(Q)<len(P)
            and ((not unit_drop) or len(P-Q)==1))

def transitive_closure(labels,A):
    idx={L:i for i,L in enumerate(labels)}; n=len(labels)
    M=[[False]*n for _ in range(n)]
    for i in range(n): M[i][i]=True
    for (P,Q),v in A.items():
        if v: M[idx[P]][idx[Q]]=True
    for k in range(n):
        for i in range(n):
            if M[i][k]:
                for j in range(n): M[i][j]=M[i][j] or M[k][j]
    Reach={L:set() for L in labels}
    for i,L in enumerate(labels):
        for j,U in enumerate(labels):
            if M[i][j]: Reach[L].add(U)
    return Reach

def all_nonempty_subsets(S: List[str]) -> List[FrozenSet[str]]:
    res=[]
    for k in range(1,len(S)+1):
        for comb in itertools.combinations(S,k): res.append(frozenset(comb))
    return res

# -------------------
# E-step: exact marginalization
# -------------------
def _phi(L, Bv):
    O = len(L & Bv)
    D = len(L - Bv)
    # Beta integral B(O+1, D+1)
    return math.exp(math.lgamma(O+1) + math.lgamma(D+1) - math.lgamma(O+D+2))

def belief_propagation(struct, root, leaf_map):
    labels = struct.labels_list
    Reach = struct.Reach
    B = compute_B_sets(root, leaf_map)
    U = {}

    # --- Upward pass (post-order) ---
    def up(v):
        for c in v.children:
            up(c)

        Bv = B[v]
        Uv = {}
        allowed = [L for L in labels if Bv.issubset(L)]
        if v.is_leaf():
            for L in allowed:
                Uv[L] = _phi(L, Bv)
        else:
            for L in allowed:
                s = 1.0
                for c in v.children:
                    Bc = B[c]
                    allowed_child = [Lc for Lc in Reach[L] if Bc.issubset(Lc)]
                    s_child = sum(U[c].get(Lc, 0) for Lc in allowed_child)
                    s *= s_child
                if s > 0:
                    Uv[L] = _phi(L, Bv) * s
        U[v] = Uv

    up(root)

    # --- Downward pass (like before) ---
    D = {}
    def down(v, Dp=None):
        Bv = B[v]
        D[v] = {L: 1.0 for L in labels if Bv.issubset(L)} if Dp is None else D[v]
        if v.is_leaf():
            return
        allowed_parent = list(D[v].keys())
        child_sums = []
        for c in v.children:
            Bc = B[c]
            Uc = U[c]
            sums = {}
            for Lp in allowed_parent:
                allowed_child = [Lc for Lc in Reach[Lp] if Bc.issubset(Lc)]
                sums[Lp] = sum(Uc.get(Lc, 0) for Lc in allowed_child)
            child_sums.append(sums)
        for idx, c in enumerate(v.children):
            Bc = B[c]
            Dc = defaultdict(float)
            for Lp in allowed_parent:
                ctx = D[v][Lp] * _phi(Lp, Bv)
                prod = 1.0
                for j, sums in enumerate(child_sums):
                    if j == idx:
                        continue
                    prod *= sums[Lp]
                for Lc in Reach[Lp]:
                    if Bc.issubset(Lc):
                        Dc[Lc] += ctx * prod
            D[c] = Dc
            down(c, D[v])

    down(root)

    # --- Marginals q ---
    q = {}
    for v, Uv in U.items():
        total = sum(Uv[L] * D.get(v, {}).get(L, 1.0) for L in Uv)
        for L, uval in Uv.items():
            val = uval * D.get(v, {}).get(L, 1.0)
            q[(id(v), L)] = val / total if total > 0 else 0.0
    return q


def aggregate_usage(q):
    usage=defaultdict(float)
    for (vid,L),p in q.items(): usage[L]+=p
    return dict(usage)

# -------------------
# Likelihood scoring
# -------------------
def beta_integral(O,D):
    return math.exp(math.lgamma(O+1)+math.lgamma(D+1)-math.lgamma(O+D+2))

def tree_loglik(struct,root,leaf_map):
    labels=struct.labels_list; Reach=struct.Reach; B=compute_B_sets(root,leaf_map)
    def dp(v,parentL=None):
        Bv=B[v]; out=defaultdict(float)
        allowed=[L for L in labels if Bv.issubset(L) and (parentL is None or L in Reach[parentL])]
        for L in allowed:
            o=len(L&Bv); d=len(L-Bv)
            if v.is_leaf():
                out[(o,d)]+=1.0
            else:
                child_tabs=[dp(c,L) for c in v.children]
                conv=child_tabs[0]
                for tab in child_tabs[1:]:
                    new=defaultdict(float)
                    for (o1,d1),w1 in conv.items():
                        for (o2,d2),w2 in tab.items():
                            new[(o1+o2,d1+d2)]+=w1*w2
                    conv=new
                for (Oc,Dc),w in conv.items():
                    out[(Oc+o,Dc+d)]+=w
        return out
    C=dp(root)
    prob=sum(w*beta_integral(O,D) for (O,D),w in C.items())
    return math.log(prob) if prob>0 else float("-inf")

def total_loglik(struct,trees,leaf_maps):
    return sum(tree_loglik(struct,t,m) for t,m in zip(trees,leaf_maps))

# -------------------
# EM loop
# -------------------
from collections import Counter

# === MILP Option B: select multis + edges with reachability via flow ===
try:
    import pulp as pl
except Exception as _e:
    pl = None

def _fmt_set(L): 
    return "{" + ",".join(sorted(L)) + "}"

def _fmt_sets(sets):
    return "[" + ", ".join(_fmt_set(s) for s in sorted(sets, key=lambda x: (len(x), tuple(sorted(x))))) + "]"

def _build_nodes_and_edges(S, candidates, unit_drop=True):
    """
    Nodes: root R, all singletons, candidate multis
    Edges: admissible subset edges among all nodes (including to/from singletons and root)
    """
    R = frozenset(S)
    singles = {frozenset([t]) for t in S}
    nodes = {R} | singles | set(candidates)

    # admissible edges under your rule
    E = []
    nodes_list = list(nodes)
    for i, P in enumerate(nodes_list):
        for Q in nodes_list:
            if admissible_edge(P, Q, unit_drop):
                E.append((P, Q))
    return nodes, E, R, singles

def wire_edges_for_selected(Z_multis, S, unit_drop=True):
    """
    Build a minimal consistent edge set:
      - parents: supersets among {root}∪multis
      - children: subsets among {multis}∪singletons
    Ensures each selected multi has >=1 parent (except root) and >=1 child.
    Greedy fallback (fast, deterministic) without solving another MILP.
    """
    root = frozenset(S)
    singles = {frozenset([t]) for t in S}
    Z = {root} | singles | set(Z_multis)

    def adm(P, Q):
        return (Q != P and Q.issubset(P) and len(Q) < len(P) and (not unit_drop or len(P - Q) == 1))

    # Candidate parents & children within Z
    parents = {L: [P for P in Z if adm(P, L)] for L in Z}
    children = {L: [Q for Q in Z if adm(L, Q)] for L in Z}

    A = {}
    # Always connect root->singles (keeps base model semantics)
    for q in singles:
        if adm(root, q):
            A[(root, q)] = 1

    # For each selected multi, ensure at least one parent and one child
    for L in Z_multis:
        # parent (prefer largest proper superset among chosen to keep shallow)
        ps = parents[L]
        if L != root:
            ps_sorted = sorted(ps, key=lambda P: (-len(P), tuple(sorted(P))))
            if ps_sorted:
                A[(ps_sorted[0], L)] = 1
        # child (prefer largest proper subset that still drops minimal elements if unit_drop True)
        cs = children[L]
        cs_sorted = sorted(cs, key=lambda Q: (-len(Q), tuple(sorted(Q))))
        if cs_sorted:
            A[(L, cs_sorted[0])] = 1

    return A


def mstep_select_multis_via_milp(usage_global, S, k, pool=None, lambda_size=1e-3, pool_cap=300):
    """
    Pick exactly k-1 multi-type potencies maximizing usage, with antichain (no A⊂B both selected).
    Returns: set of frozensets (the selected potencies).
    """
    root = frozenset(S)

    # --- Candidate pool C ---
    # If 'pool' supplied (e.g., from harvest_internal_multis), use it; else derive from usage_global.
    if pool is None:
        C_full = {L for L in usage_global if len(L) >= 2 and L != root}
    else:
        C_full = {L for L in pool if len(L) >= 2 and L != root}

    # Rank by usage and cap pool size for speed
    ranked = sorted(C_full, key=lambda L: (usage_global.get(L, 0.0), -len(L), tuple(sorted(L))), reverse=True)
    C = ranked[:min(len(ranked), max(0, pool_cap))]

    # If pool is too small, bail early
    need = max(0, k - 1)
    if len(C) < need:
        # pad with pairs to have at least k-1 candidates
        extra_pairs = [frozenset(p) for p in itertools.combinations(S, 2)]
        for P in extra_pairs:
            if P not in C and P != root:
                C.append(P)
                if len(C) >= need:
                    break

    # --- Build MILP ---
    prob = pl.LpProblem("Mstep_SelectMultis", pl.LpMaximize)

    y = {L: pl.LpVariable(f"y_{hash(L)}", lowBound=0, upBound=1, cat=pl.LpBinary) for L in C}

    # Objective: usage - lambda * size
    prob += pl.lpSum((usage_global.get(L, 0.0) - lambda_size * len(L)) * y[L] for L in C)

    # Exact budget: pick exactly k-1 multis
    prob += pl.lpSum(y[L] for L in C) == max(0, k - 1), "budget_k_minus_1"

    # Antichain constraints: y[A] + y[B] <= 1 if A ⊂ B
    # (O(|C|^2) worst case; OK for a capped pool)
    C_set = set(C)
    C_sorted = sorted(C, key=lambda L: (len(L), tuple(sorted(L))))
    for i in range(len(C_sorted)):
        A = C_sorted[i]
        for j in range(i + 1, len(C_sorted)):
            B = C_sorted[j]
            # quick reject via sizes
            if len(A) >= len(B):
                continue
            if A.issubset(B):
                prob += y[A] + y[B] <= 1, f"antichain_{hash((A,B))}"

    # Solve (use CBC; can swap to GUROBI_CMD if available)
    try:
        solver = pl.GUROBI_CMD(msg=True, timeLimit=60)
    except Exception:
        solver = pl.PULP_CBC_CMD(msg=True, timeLimit=60)
    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved", "Infeasible", "Feasible"):
        print(f"[MILP] unusual status: {status}")

    keep = {L for L in C if pl.value(y[L]) >= 0.5}
    # In case of Feasible (not Optimal), enforce size exactly:
    if len(keep) != need:
        # fallback: take top 'need' by usage among those picked or relax
        chosen = sorted(C, key=lambda L: (usage_global.get(L, 0.0), -len(L)), reverse=True)[:need]
        keep = set(chosen)

    return keep


def _solve_mstep_milp_optionB(S, candidates, k, weights, unit_drop=True, time_limit=None, msg=False):
    """
    Solve the Option-B MILP:
      - select k-1 multis (y_L)
      - choose edges x_(P,Q)
      - enforce reachability via one-commodity flow to a dummy sink through singletons
    Returns:
      keep_multis (set of frozenset), chosen_edges (set of (P,Q))
    """
    if pl is None:
        raise RuntimeError("pulp not available. Install with: pip install pulp")

    nodes, E, R, singles = _build_nodes_and_edges(S, candidates, unit_drop=unit_drop)
    C = set(candidates)  # only multis are decision y_L; root/singles are always active

    # --- Model ---
    m = pl.LpProblem("mstep_optionB", pl.LpMaximize)

    # --- Vars ---
    # activity
    y = {L: pl.LpVariable(f"y_{hash(L)}", lowBound=0, upBound=1, cat="Binary") for L in C}
    # edges
    x = {(P, Q): pl.LpVariable(f"x_{hash(P)}_{hash(Q)}", lowBound=0, upBound=1, cat="Binary") for (P, Q) in E}
    # flow
    f = {(P, Q): pl.LpVariable(f"f_{hash(P)}_{hash(Q)}", lowBound=0) for (P, Q) in E}
    # edges from each singleton to dummy sink T (absorb flow)
    # implement T as a special key None; only flow vars, no x for these arcs
    f_to_T = {u: pl.LpVariable(f"f_{hash(u)}_T", lowBound=0) for u in singles}

    # active indicator a_U for all nodes (root/singletons fixed to 1, multis -> y_L)
    def a(U):
        if U == R: return 1.0
        if len(U) == 1: return 1.0
        return y[U]

    # --- Objective: maximize sum weights of selected multis ---
    w = {L: float(weights.get(L, 0.0)) for L in C}
    m += pl.lpSum(w[L] * y[L] for L in C)

    # --- Constraints ---

    # budget: exactly k-1 multis
    m += pl.lpSum(y[L] for L in C) == max(0, k - 1), "budget_k_minus_1"

    # edges allowed only if endpoints active
    for (P, Q) in E:
        m += x[(P, Q)] <= a(P), f"edge_endpoint_active_src_{hash(P)}_{hash(Q)}"
        m += x[(P, Q)] <= a(Q), f"edge_endpoint_active_dst_{hash(P)}_{hash(Q)}"

    # degree constraints: each selected multi has >=1 parent and >=1 child
    for L in C:
        in_edges  = [x[(P, L)] for (P, Q) in E if Q == L]
        out_edges = [x[(L, Q)] for (P, Q) in E if P == L]
        if in_edges:
            m += pl.lpSum(in_edges) >= y[L], f"parent_exists_{hash(L)}"
        else:
            # if no admissible parents exist (shouldn't happen under subset rule), forbid selection
            m += y[L] == 0, f"no_parents_forbid_{hash(L)}"
        if out_edges:
            m += pl.lpSum(out_edges) >= y[L], f"child_exists_{hash(L)}"
        else:
            m += y[L] == 0, f"no_children_forbid_{hash(L)}"

    # flow capacity: f_(P,Q) <= x_(P,Q)
    for (P, Q) in E:
        m += f[(P, Q)] <= x[(P, Q)], f"flow_cap_{hash(P)}_{hash(Q)}"

    # root supply equals number of selected multis
    # flow balance at root: out - in = sum y
    m += (
        pl.lpSum(f[(R, Q)] for (P, Q) in E if P == R) -
        pl.lpSum(f[(P, R)] for (P, Q) in E if Q == R)
        == pl.lpSum(y[L] for L in C)
    ), "root_supply"

    # singletons can dump flow to T; no x-var needed there (they're always active)
    # add conservation at singletons: (in + from parent edges) - (out + to T) == 0
    for u in singles:
        m += (
            pl.lpSum(f[(P, u)] for (P, Q) in E if Q == u) -
            pl.lpSum(f[(u, Q)] for (P, Q) in E if P == u) -
            f_to_T[u]
            == 0
        ), f"balance_singleton_{hash(u)}"

    # All flow ends at T: sum to T equals total supply
    m += pl.lpSum(f_to_T[u] for u in singles) == pl.lpSum(y[L] for L in C), "sink_absorbs_all"

    # every selected multi must receive >= 1 unit from the root side (reachability)
    for L in C:
        m += (
            pl.lpSum(f[(P, L)] for (P, Q) in E if Q == L) -
            pl.lpSum(f[(L, Q)] for (P, Q) in E if P == L)
            >= y[L]
        ), f"reachability_{hash(L)}"

    # Flow conservation at all other nodes (non-root, non-singleton)
    for U in nodes:
        if U == R or U in singles:
            continue
        # multis: we already put ">= y[L]" constraint (so we *allow* accumulation).
        # To keep balances tidy, enforce net inflow - outflow >= 0 (can accumulate), which is implied by >= y[L] anyway.
        # For safety, add a weak non-negativity:
        m += (
            pl.lpSum(f[(P, U)] for (P, Q) in E if Q == U) -
            pl.lpSum(f[(U, Q)] for (P, Q) in E if P == U)
            >= 0
        ), f"nonneg_balance_{hash(U)}"

    # --- Solve ---
    if time_limit is not None:
        m.solve(pl.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    else:
        m.solve(pl.PULP_CBC_CMD(msg=msg))

    if pl.LpStatus[m.status] not in ("Optimal", "Not Solved", "Undefined", "Infeasible"):
        # CBC sometimes returns "Integer Feasible" — treat as OK
        pass

    # --- Extract solution ---
    keep_multis = {L for L in C if y[L].varValue and y[L].varValue > 0.5}
    chosen_edges = {(P, Q) for (P, Q) in E if x[(P, Q)].varValue and x[(P, Q)].varValue > 0.5}

    return keep_multis, chosen_edges

def _structure_from_solution(S, keep_multis, chosen_edges, unit_drop=True):
    R = frozenset(S)
    singles = {frozenset([t]) for t in S}
    Z = {R} | singles | set(keep_multis)
    # only edges among active nodes
    active = Z
    A = {(P, Q): 1 for (P, Q) in chosen_edges if (P in active and Q in active)}
    # Safety: if some selected multi ended up with no edges (shouldn’t happen), wire minimal edges
    for L in keep_multis:
        has_in  = any((P, Q) in A and Q == L for (P, Q) in A)
        has_out = any((P, Q) in A and P == L for (P, Q) in A)
        if not has_in:
            # attach an admissible parent; prefer R or any superset
            parents = [P for P in active if admissible_edge(P, L, unit_drop)]
            if parents:
                A[(parents[0], L)] = 1
        if not has_out:
            # attach an admissible child; prefer a singleton
            children = [Q for Q in active if admissible_edge(L, Q, unit_drop)]
            if children:
                A[(L, children[0])] = 1
    return Structure(S, Z, A, unit_drop=unit_drop)


def harvest_internal_multis(trees, leaf_maps, include_subsets=True, top_m=None):
    """
    Mine multi-type potency candidates from INTERNAL nodes only.

    For each tree:
      - Compute B(v) for every node v.
      - For internal nodes with |B(v)| >= 2:
          * if include_subsets=True: count ALL subsets of B(v) (size >= 2)
          * else: count the exact B(v) only
    Return: a set of frozensets (top_m by frequency; tie-break by size then lex).
    """
    cand = Counter()
    for root, lmap in zip(trees, leaf_maps):
        B = compute_B_sets(root, lmap)
        stack = [root]
        while stack:
            v = stack.pop()
            stack.extend(v.children)
            if v.is_leaf():
                continue
            bv = B[v]
            if len(bv) < 2:
                continue
            if include_subsets:
                bl = sorted(bv)
                for r in range(2, len(bl) + 1):
                    for comb in itertools.combinations(bl, r):
                        cand[frozenset(comb)] += 1
            else:
                cand[frozenset(bv)] += 1

    ranked = sorted(
        cand.items(),
        key=lambda kv: (kv[1], len(kv[0]), tuple(sorted(kv[0]))),
        reverse=True
    )
    keep = [k for k, _ in ranked]
    if top_m is not None:
        keep = keep[:max(0, top_m)]
    return set(keep)



# def em_infer(S, trees, leaf_maps, k, iters=100, p=0.8):
#     def _fmt_set(L): return "{" + ",".join(sorted(L)) + "}"
#     def _fmt_sets(sets):
#         return "[" + ", ".join(_fmt_set(s) for s in sorted(sets, key=lambda x: (len(x), tuple(sorted(x))))) + "]"

#     root = frozenset(S)
#     singles = {frozenset([t]) for t in S}

#     # --- SEED multis from internal nodes (your earlier strategy)
#     seeded = harvest_internal_multis(
#         trees, leaf_maps,
#         include_subsets=True,           # ← same behavior as before
#         top_m=max(0, k - 1)
#     )
#     # Fallback padding if harvesting returns fewer than needed
#     if len(seeded) < max(0, k - 1):
#         rng = random.Random(0)
#         pool = [frozenset(x) for x in itertools.combinations(S, 2)]
#         rng.shuffle(pool)
#         for P in pool:
#             if P not in seeded:
#                 seeded.add(P)
#                 if len(seeded) >= max(0, k - 1):
#                     break

#     Z0 = {root} | singles | seeded
#     A0 = {(root, q): 1 for q in singles}
#     struct = Structure(S, Z0, A0)

#     prev_keep = set(seeded)

#     for it in range(1, iters + 1):
#         # ------- E-step -------
#         usage_global = defaultdict(float)
#         for tree, lmap in zip(trees, leaf_maps):
#             q = belief_propagation(struct, tree, lmap)
#             for L, u in aggregate_usage(q).items():
#                 usage_global[L] += u

#         # ------- M-step -------
#         multis = [L for L in usage_global if len(L) >= 2 and L != root]
#         multis.sort(key=lambda L: (usage_global[L], len(L), tuple(sorted(L))), reverse=True)
#         keep = set(multis[:max(0, k - 1)])

#         # Print chosen potencies & alert on diffs
#         print(f"[EM] iter {it:03d}  chosen_multis ({len(keep)}): {_fmt_sets(keep)}")
#         if it > 1:
#             added = keep - prev_keep
#             removed = prev_keep - keep
#             if added or removed:
#                 parts = []
#                 if added: parts.append(f"ADDED: {_fmt_sets(added)}")
#                 if removed: parts.append(f"REMOVED: {_fmt_sets(removed)}")
#                 print("   >>> ALERT: potency set changed — " + " | ".join(parts))

#         # Rebuild structure for next iter
#         Znew = {root} | singles | keep
#         Anew = {(root, q): 1 for q in singles}
#         struct = Structure(S, Znew, Anew)

#         prev_keep = keep

#     return struct

def em_infer(S, trees, leaf_maps, k, iters=100, unit_drop=True, time_limit=None, milp_verbose=False):
    """
    EM with MILP (Option B) in the M-step:
        E-step: belief propagation -> q(v,L) -> weights w_L
        M-step: MILP to select k-1 multis AND edges with reachability
    """
    rng = random.Random(0)
    R = frozenset(S)
    singles = {frozenset([t]) for t in S}

    # ---- Candidate pool (broad but relevant) ----
    #   Use your internal-node miner; keep a healthy superset (>= k-1)
    cand_pool = harvest_internal_multis(trees, leaf_maps, include_subsets=True, top_m=None)
    # remove degenerate/singletons/root
    cand_pool = {L for L in cand_pool if len(L) >= 2 and L != R}
    if not cand_pool:
        # fallback to all pairs
        cand_pool = {frozenset(x) for x in itertools.combinations(S, 2)}

    # ---- Start with star ----
    Z0 = {R} | singles
    A0 = {(R, q): 1 for q in singles}
    struct = Structure(S, Z0, A0, unit_drop=unit_drop)

    prev_keep = set()
    prev_edges = set()

    for it in range(1, iters + 1):
        # ===== E-step =====
        usage_global = defaultdict(float)
        for tree, lmap in zip(trees, leaf_maps):
            q = belief_propagation(struct, tree, lmap)
            for L, u in aggregate_usage(q).items():
                usage_global[L] += u

        # Restrict weights to candidate multis only
        weights = {L: usage_global.get(L, 0.0) for L in cand_pool}

        # ===== M-step: MILP (Option B) =====
        keep_multis, chosen_edges = _solve_mstep_milp_optionB(
            S=S,
            candidates=cand_pool,
            k=k,
            weights=weights,
            unit_drop=unit_drop,
            time_limit=time_limit,
            msg=milp_verbose
        )

        # Debug/print selection
        print(f"[EM] iter {it:03d}  chosen_multis ({len(keep_multis)}): {_fmt_sets(keep_multis)}")

        # Alert on set diffs
        if it > 1:
            added = keep_multis - prev_keep
            removed = prev_keep - keep_multis
            if added or removed:
                parts = []
                if added:   parts.append(f"ADDED: {_fmt_sets(added)}")
                if removed: parts.append(f"REMOVED: {_fmt_sets(removed)}")
                print("   >>> ALERT (sets): " + " | ".join(parts))

        # Alert on edge diffs (compact)
        edge_set = set(chosen_edges)
        if it > 1:
            e_added  = edge_set - prev_edges
            e_removed = prev_edges - edge_set
            if e_added or e_removed:
                def _fmt_edge(e): return f"{_fmt_set(e[0])}->{_fmt_set(e[1])}"
                a_str = ", ".join(_fmt_edge(e) for e in sorted(e_added, key=lambda uv: (len(uv[0]), len(uv[1]))))
                r_str = ", ".join(_fmt_edge(e) for e in sorted(e_removed, key=lambda uv: (len(uv[0]), len(uv[1]))))
                print("   >>> ALERT (edges): " + (f"ADDED: [{a_str}] " if e_added else "") + (f"REMOVED: [{r_str}]" if e_removed else ""))

        # Build next structure from MILP
        struct = _structure_from_solution(S, keep_multis, edge_set, unit_drop=unit_drop)

        prev_keep = set(keep_multis)
        prev_edges = set(edge_set)

    return struct


# -------------------
# Input + GT loader (unchanged style)
# -------------------
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

def filter_leaf_map_to_tree(root, leaf_map): 
    leaves=set(collect_leaf_names(root))
    return {l:leaf_map[l] for l in leaves if l in leaf_map}

def read_trees_and_maps(tree_paths, meta_paths):
    trees = [read_newick_file(p) for p in tree_paths]
    raw   = [read_leaf_type_map(p) for p in meta_paths]
    maps  = [filter_leaf_map_to_tree(t, m) for t, m in zip(trees, raw)]
    S     = sorted({str(v) for m in maps for v in m.values()})

    # --- DEBUG: alignment between trees and meta ---
    # print("\n[DEBUG] read_trees_and_maps — alignment check")
    # for i, (t, m_raw, m_filt) in enumerate(zip(trees, raw, maps)):
    #     leaves = collect_leaf_names(t)
    #     leaves_set = set(leaves)
    #     mapped_in_raw  = len(leaves_set & set(m_raw.keys()))
    #     mapped_in_filt = len(m_filt)
    #     missing = [x for x in leaves if x not in m_raw][:5]
    #     # print(f"  Tree {i}: leaves={len(leaves)}, in_raw={mapped_in_raw}, in_filtered={mapped_in_filt}")
    #     if missing:
    #         # print(f"    e.g. missing in meta: {missing[:5]}")
    #     # quick peek at some mappings
    #     peek = list(m_filt.items())[:5]
    #     if peek:
    #         # print(f"    sample mappings: {peek}")
    #     else:
    #         # print(f"    sample m/appings: <none>")

    # print(f"[DEBUG] S (types from meta across all trees): {S}")
    return trees, maps, S


def _resolve_id_to_set(id_str, comp_map, memo):
    id_str=str(id_str)
    if id_str.startswith("-"): return frozenset([id_str])
    if id_str in memo: return memo[id_str]
    if id_str not in comp_map: raise ValueError(f"Missing comp {id_str}")
    val=comp_map[id_str]; acc=set()
    if isinstance(val,list):
        for c in val: acc|=_resolve_id_to_set(c,comp_map,memo)
    else: acc|=_resolve_id_to_set(val,comp_map,memo)
    memo[id_str]=frozenset(acc); return memo[id_str]

def build_ground_truth(txt_path, unit_drop=False):
    objs=[json.loads(line) for line in open(txt_path) if line.strip()]
    adj=objs[0]; comp_map=objs[2]

    # --- FIX: drop synthetic "root" if present ---
    if "root" in adj:
        ch = adj["root"]
        if isinstance(ch,list) and len(ch)==1:
            adj = {k:v for k,v in adj.items() if k!="root"}
        else:
            raise ValueError("Synthetic root in GT file not in expected format")

    memo={}; pot_map={}
    for idv in set(comp_map.keys())|set(adj.keys()):
        if str(idv).startswith("-"):
            memo[str(idv)]=frozenset([str(idv)])
        else:
            pot_map[idv]=_resolve_id_to_set(idv,comp_map,memo)
    Z={frozenset([t]) for s in memo.values() for t in s if t.startswith("-")}
    for s in pot_map.values():
        if len(s)>=2: Z.add(s)
    A={}
    for u,chs in adj.items():
        Pu=pot_map.get(u,frozenset([u]))
        for v in chs:
            Qv=pot_map.get(v,frozenset([v]))
            if admissible_edge(Pu,Qv,unit_drop): A[(Pu,Qv)]=1
    return Z,A


def jaccard_distance(set1,set2):
    return 1 - len(set1&set2)/len(set1|set2) if set1 or set2 else 0.0

# -------------------
# Paths (exact same style as before)
# -------------------
def build_fate_map_path(map_idx, type_num, tree_kind="graph"):
    idx4=f"{map_idx:04d}"
    return os.path.join("inputs","differentiation_maps",tree_kind,f"type_{type_num}",f"graph_fate_map{idx4}.txt"), idx4

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


def pretty_print_sets(name, sets):
    print(f"\n{name}:")
    for s in sorted(sets, key=lambda x: (len(x), sorted(x))):
        print("  ", sorted(list(s)))


# -------------------
# Runner (drop-in replacement)
# -------------------
def process_case(map_idx, type_num, cells_n, k, tree_kind="graph"):
    fate_map_path, idx4=build_fate_map_path(map_idx,type_num,tree_kind)
    tree_paths, meta_paths=build_tree_and_meta_paths(map_idx,type_num,cells_n,tree_kind)
    trees,leaf_maps,S=read_trees_and_maps(tree_paths,meta_paths)
    struct=em_infer(S,trees,leaf_maps,k,iters=100)

    # --- Predicted loss (EXACTLY like the old code computes it) ---
    # Use the same scorer and Priors; pred_loss is the sum of per-tree log-likelihoods.
    dummy_priors = Priors(potency_mode="fixed_k", fixed_k= k, rho=0.2)
    pred_logpost, per_tree_logs_pred = score_structure(
        struct=struct,
        trees=trees,
        leaf_type_maps=leaf_maps,
        priors=dummy_priors,
        prune_eps=0.0
    )
    pred_loss = sum(per_tree_logs_pred)  # this is what the old code prints as "Pred loss"

    # --- Ground-truth loss (EXACTLY like before) ---
    gt_sets, gt_loss = score_given_map_and_trees(
        fate_map_path, trees, meta_paths, fixed_k= k, unit_drop_edges=False
    )

    # --- Jaccard on potencies (same as before) ---
    predicted_sets = {P for P in struct.Z_active if len(P) >= 2}
    jd = jaccard_distance(predicted_sets, gt_sets)

    # --- print / return (unchanged formatting) ---
    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", gt_sets)
    print("\n=== Jaccard Distance ===")
    print(f"Jaccard Distance (Pred vs GT): {jd:.6f}")
    print(f"Predicted map's loss: {pred_loss:.6f}")
    print(f"Ground truth's loss: {gt_loss:.6f}")

    return jd, gt_loss, pred_loss


    # predicted={P for P in struct.Z_active if len(P)>=2}
    # gt_sets,gt_A=build_ground_truth(fate_map_path)
    # jd=jaccard_distance(predicted,gt_sets)
    # pred_ll=total_loglik(struct,trees,leaf_maps)
    # gt_struct=Structure(S,gt_sets,gt_A); gt_ll=total_loglik(gt_struct,trees,leaf_maps)
    # print(f"\n=== type_{type_num} map {idx4} cells_{cells_n} ===")
    # print("Predicted sets:"); [print(" ",sorted(P)) for P in predicted]
    # print("Ground truth sets:"); [print(" ",sorted(P)) for P in gt_sets]
    # print("Jaccard:",jd)
    # print("Pred loss:",-pred_ll)
    # print("GT loss:",-gt_ll)
    # return jd,-gt_ll,-pred_ll


def main_multi(type_nums=[10], maps_start=26, maps_end=26, cells_list=[50], fixed_k=5, tree_kind="graph"):
    results=[]
    for t in type_nums:
        for idx in range(maps_start,maps_end+1):
            for cells in cells_list:
                try:
                    jd,gt_loss,pred_loss=process_case(idx,t,cells,fixed_k,tree_kind)
                    results.append((t,idx,cells,jd,gt_loss,pred_loss))
                except Exception as e:
                    print(f"[ERROR] {e}")
                    traceback.print_exc()
    print("\n=== Summary ===")
    for t,idx,cells,jd,gt,pred in results:
        print(f"type {t}, map {idx}, cells {cells}: Jaccard={jd:.3f}, GT loss={gt:.3f}, Pred loss={pred:.3f}")

if __name__=="__main__":
    main_multi(type_nums=[6], 
               maps_start=26, 
               maps_end=26, 
               cells_list=[50], 
               fixed_k=5,
               tree_kind = "graph")