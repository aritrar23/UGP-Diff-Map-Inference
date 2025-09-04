#!/usr/bin/env python3
"""
Run CARTA on the new inputs/ layout and compute Jaccard vs ground truth.

Layout used (only poly_tree):
inputs/
  trees/
    poly_tree/
      type_8/ type_12/ type_16/
        cells_50/ cells_100/ cells_200/
          0002_tree_0.txt
          0002_meta_0.txt
          ...
  differentiation_maps/
    poly_tree/
      type_8/ type_12/ type_16/
        graph_fate_map0002.txt
        graph_fate_map0004.txt
        ...

We detect datasets by ID in filenames under trees/.../cells_*:
  r'^(\d{4})_(tree|meta)_(\d+)\.txt$'

By default, k is forced from the type folder name:
  type_8  -> k = 7
  type_12 -> k = 11
  type_16 -> k = 15
Pass --multi_k with -k ... to override and try multiple k's.
"""

import os
import re
import csv
import glob
import json
import argparse
import subprocess
from typing import List, Set, FrozenSet, Tuple, Optional, Dict

# ----------------- token canonicalization -----------------

def _canon_token(x: str) -> str:
    x = str(x).strip()
    while len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'":
        x = x[1:-1].strip()
    return x

# ----------------- CARTA input builders -----------------

def write_file_locations(newicks: List[str], metas: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for n, m in zip(newicks, metas):
            f.write(f"{n}\t{m}\n")  # TAB!

def _states_from_meta(meta_path: str) -> Set[str]:
    out=set()
    with open(meta_path, "r", encoding="utf-8") as f:
        first=f.readline()
        if not first: return out
        delim = "\t" if ("\t" in first or meta_path.endswith((".tsv",".txt"))) else ","
        hdr=[c.strip().lower() for c in first.strip().split(delim)]
        has_header = len(hdr)>=2 and ("type" in hdr[1] or "cell_state" in hdr[1])
        f.seek(0)
        for i, line in enumerate(f,1):
            if has_header and i==1: continue
            parts=[c.strip() for c in line.strip().split(delim)]
            if len(parts)>=2 and parts[1]!="":
                out.add(_canon_token(parts[1]))
    return out

def write_states_file(metas: List[str], out_path: str) -> List[str]:
    S=set()
    for p in metas: S |= _states_from_meta(p)
    S=sorted(S)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in S: f.write(s + "\n")
    return S

# ----------------- parse CARTA outputs -----------------

def _parse_set_token(line: str) -> Optional[FrozenSet[str]]:
    l = line.strip()
    if not l or l.startswith("#"): return None
    if l.startswith("[") and l.endswith("]"):
        try:
            arr = json.loads(l)
            items = [_canon_token(x) for x in arr]
            return frozenset(items) if len(items) >= 2 else None
        except Exception:
            pass
    if l.startswith("{") and l.endswith("}"):
        inside = l[1:-1].strip()
        items = [_canon_token(t) for t in inside.replace(",", " ").split() if t]
        return frozenset(items) if len(items) >= 2 else None
    toks = [_canon_token(t) for t in l.replace(",", " ").split() if t]
    return frozenset(toks) if len(toks) >= 2 else None

def parse_progenitors(prog_file: str, states: List[str]) -> Set[FrozenSet[str]]:
    out = set()
    S = [_canon_token(s) for s in states]
    with open(prog_file, "r", encoding="utf-8") as f:
        for line in f:
            s = _parse_set_token(line)
            if s:
                out.add(s)
                continue
            toks = [t for t in line.strip().replace(",", " ").split() if t]
            if toks and all(t in ("0", "1") for t in toks) and len(toks) == len(S):
                chosen = [S[i] for i, t in enumerate(toks) if t == "1"]
                if len(chosen) >= 2:
                    out.add(frozenset(chosen))
    return out

# ----------------- parse ground truth maps -----------------

def _read_json_objects_exact(path: str) -> List[object]:
    objs=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                objs.append(json.loads(ln))
    if not objs:
        raise ValueError(f"{path}: no JSON objects found")
    return objs

def _normalize_adj_remove_synthetic_root(adj: dict) -> dict:
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if isinstance(ch, list) and len(ch) == 1:
            del adj2["root"]
    return adj2

def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set) -> frozenset:
    id_str = _canon_token(id_str)
    if id_str.startswith("-"):
        return frozenset([id_str])
    if id_str in memo:
        return memo[id_str]
    if id_str in visiting:
        raise ValueError(f"Cycle detected while resolving '{id_str}'")
    if id_str not in comp_map:
        raise ValueError(f"Id '{id_str}' appears in adjacency but is missing in composition map")
    visiting.add(id_str)
    val = comp_map[id_str]
    acc = set()
    if isinstance(val, list):
        for child in val:
            acc |= _resolve_id_to_set(child, comp_map, memo, visiting)
    else:
        acc |= _resolve_id_to_set(val, comp_map, memo, visiting)
    visiting.remove(id_str)
    memo[id_str] = frozenset(_canon_token(x) for x in acc)
    return memo[id_str]

def ground_truth_sets(map_txt: str) -> Set[FrozenSet[str]]:
    """
    Ground truth = multi-type potencies that appear as nodes in the adjacency,
    expanded via the composition map. Singletons excluded.
    """
    objs = _read_json_objects_exact(map_txt)
    # adjacency: first dict-of-lists
    adj = None
    for o in objs:
        if isinstance(o, dict) and any(isinstance(v, list) for v in o.values()):
            adj = {str(k): [str(x) for x in v] for k, v in o.items() if isinstance(v, list)}
            break
    if adj is None:
        raise ValueError(f"{map_txt}: could not locate adjacency dict")
    comp_map = objs[2] if len(objs) >= 3 and isinstance(objs[2], dict) else None
    if comp_map is None:
        raise ValueError(f"{map_txt}: cannot find composition map (3rd JSON)")
    comp_map = {_canon_token(k): v for k, v in comp_map.items()}

    adj = _normalize_adj_remove_synthetic_root(adj)

    ids = set(adj.keys())
    for chs in adj.values():
        for v in chs:
            ids.add(str(v))

    memo={}
    gt=set()
    for vid in ids:
        vid = _canon_token(vid)
        if vid.startswith("-"):
            continue
        s = _resolve_id_to_set(vid, comp_map, memo, visiting=set())
        if len(s) >= 2:
            gt.add(s)
    return gt

# ----------------- Jaccard -----------------

def jaccard(A:Set[FrozenSet[str]], B:Set[FrozenSet[str]])->float:
    if not A and not B: return 0.0
    return 1.0 - (len(A & B) / len(A | B) if (A or B) else 0.0)

# ----------------- CARTA runner (multi-k) -----------------

def _extract_k_from_filename(path: str) -> Optional[int]:
    b = os.path.basename(path)
    m = re.search(r"progenitors[_-](\d+)\.txt$", b)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\.txt$", b)
    return int(m.group(1)) if m else None

def run_carta(prefix: str, newicks: List[str], metas: List[str], states_file: str,
              klist: List[int], normalize_method: str, time_limit_sec: Optional[int],
              enforce_tree: bool, mode: str, progen_matrix: Optional[str]) -> Tuple[Dict[int,str], str]:
    loc_file = prefix + "_file_locations.tsv"
    stdout_file = prefix + "_stdout.txt"
    write_file_locations(newicks, metas, loc_file)

    cmd = [
        "carta",
        "--prefix", prefix,
        "--file_locations", loc_file,
        "--states_file", states_file,
        "--normalize_method", normalize_method,
        "-m", mode
    ]
    if enforce_tree: cmd += ["--enforce_tree"]
    if time_limit_sec is not None: cmd += ["--time_limit_sec", str(time_limit_sec)]
    if klist: cmd += ["--klist", *map(str, klist)]
    if progen_matrix: cmd += ["--progen_matrix", progen_matrix]

    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n" + (proc.stdout or "") + "\n\n=== STDERR ===\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        raise RuntimeError(f"CARTA failed (code {proc.returncode}). See {stdout_file}.")

    base = os.path.dirname(prefix) or "."
    stem = os.path.basename(prefix)
    cands = sorted(set(glob.glob(os.path.join(base, f"{stem}*progenitors*.txt"))))
    if not cands:
        raise FileNotFoundError(f"No progenitors files produced for '{prefix}'. Check {stdout_file}.")

    k2file: Dict[int,str] = {}
    for p in cands:
        k = _extract_k_from_filename(p)
        if k is None: continue
        if k not in k2file or os.path.getmtime(p) > os.path.getmtime(k2file[k]):
            k2file[k] = p
    if not k2file:
        raise ValueError(f"Could not associate any progenitors file with a k for prefix '{prefix}'.")
    return k2file, stdout_file

# ----------------- discovery in inputs/ layout -----------------

ID_RE = re.compile(r"^(\d{4})_(tree|meta)_(\d+)\.txt$")

def discover_ids_and_files(trees_cells_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Return {id: (tree_files, meta_files)} for files matching our pattern.
    Ignores anything else (e.g., *_cm_*).
    """
    id2trees: Dict[str,List[str]] = {}
    id2metas: Dict[str,List[str]] = {}
    for path in glob.glob(os.path.join(trees_cells_dir, "*.txt")):
        base = os.path.basename(path)
        m = ID_RE.match(base)
        if not m:
            continue
        idd, kind, _ = m.groups()
        if kind == "tree":
            id2trees.setdefault(idd, []).append(path)
        elif kind == "meta":
            id2metas.setdefault(idd, []).append(path)
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for idd in sorted(set(id2trees) | set(id2metas)):
        trees = sorted(id2trees.get(idd, []))
        metas = sorted(id2metas.get(idd, []))
        if trees and metas and len(trees) == len(metas):
            out[idd] = (trees, metas)
    return out

def map_file_for_id(maps_type_dir: str, idd: str) -> str:
    cand = os.path.join(maps_type_dir, f"graph_fate_map{idd}.txt")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"Missing ground-truth map for id {idd}: {cand}")
    return cand

# ----------------- per-dataset processing -----------------

def process_dataset(prefix: str,
                    trees: List[str],
                    metas: List[str],
                    map_txt: str,
                    klist: List[int],
                    normalize_method: str,
                    time_limit_sec: Optional[int],
                    enforce_tree: bool,
                    mode: str,
                    progen_matrix: Optional[str]) -> Tuple[float, int]:
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    # states
    states_file = prefix + "_states.txt"
    states = write_states_file(metas, states_file)

    # run CARTA
    k2file, _ = run_carta(prefix, trees, metas, states_file, klist,
                          normalize_method, time_limit_sec, enforce_tree, mode, progen_matrix)
    gt_sets = ground_truth_sets(map_txt)

    # evaluate all ks and pick best
    best_k, best_jd = None, None
    for k in sorted(k2file):
        pred_sets = parse_progenitors(k2file[k], states)
        jd = jaccard(pred_sets, gt_sets)
        print(f"  k={k:<3d}  Jaccard={jd:.6f}  file={os.path.basename(k2file[k])}")
        if best_jd is None or jd < best_jd:
            best_jd, best_k = jd, k
    return best_jd if best_jd is not None else 1.0, best_k if best_k is not None else -1

# ----------------- k from type folder -----------------

def k_from_type(typ: str) -> int:
    m = re.search(r"(\d+)$", typ)
    if not m:
        raise ValueError(f"Cannot parse k from type dir '{typ}' (expected like 'type_12').")
    k = int(m.group(1)) - 1
    if k < 1:
        raise ValueError(f"Derived k={k} from '{typ}' is invalid.")
    return k

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Run CARTA on inputs/trees & differentiation_maps (poly_tree only).")
    ap.add_argument("--inputs-root", default="inputs", help="Root containing trees/ and differentiation_maps/")
    ap.add_argument("--types", nargs="+", default=["type_8"], help="Type dirs under poly_tree")
    ap.add_argument("--cells", nargs="+", default=["cells_50","cells_100"], help="Cells dirs under each type")
    ap.add_argument("-k", "--klist", nargs="+", type=int, default=[5], help="k values (used only with --multi_k)")
    ap.add_argument("--multi_k", action="store_true",
                    help="Use --klist exactly; otherwise force k = (number in type_X) - 1 per type.")
    ap.add_argument("--normalize_method",
                    choices=["no_normalization","cell_proportion_before_pruning","cell_proportion_after_pruning"],
                    default="no_normalization")
    ap.add_argument("--time_limit_sec", type=int, default=300)
    ap.add_argument("--no_enforce_tree", action="store_true")
    ap.add_argument("-m", "--mode", choices=["exact","round"], default="exact")
    ap.add_argument("--progen_matrix", help="Optional CSV putative progenitors.")
    ap.add_argument("--out-root", default="carta_runs_inputs")
    ap.add_argument("--csv", default="carta_runs_inputs/summary.csv")
    args = ap.parse_args()

    trees_root = os.path.join(args.inputs_root, "trees", "bin_tree")
    maps_root  = os.path.join(args.inputs_root, "differentiation_maps", "bin_tree")

    results = []  # rows for CSV

    for typ in args.types:
        trees_type_dir = os.path.join(trees_root, typ)
        maps_type_dir  = os.path.join(maps_root, typ)
        if not os.path.isdir(trees_type_dir) or not os.path.isdir(maps_type_dir):
            print(f"[skip] missing type dir: {typ}")
            continue

        # choose k for this type
        klist_eff = args.klist if args.multi_k else [k_from_type(typ)]
        print(f"[k] using klist {klist_eff} for {typ} "
              f"({'override (--multi_k)' if args.multi_k else 'derived from type'})")

        for cells in args.cells:
            cells_dir = os.path.join(trees_type_dir, cells)
            if not os.path.isdir(cells_dir):
                print(f"[skip] {typ}/{cells} not found")
                continue

            print(f"\n=== Processing {typ}/{cells} ===")
            id2files = discover_ids_and_files(cells_dir)
            if not id2files:
                print(f"[warn] no (id_tree_i.txt, id_meta_i.txt) pairs found in {cells_dir}")
                continue

            for idd, (trees, metas) in id2files.items():
                try:
                    gt_map = map_file_for_id(maps_type_dir, idd)
                except FileNotFoundError as e:
                    print(f"[skip id {idd}] {e}")
                    continue

                prefix = os.path.join(args.out_root, typ, cells, idd, f"{typ}_{cells}_{idd}")
                print(f"- id={idd}  (pairs={len(trees)})  map={os.path.basename(gt_map)}")
                try:
                    jd, best_k = process_dataset(prefix, trees, metas, gt_map,
                                                 klist_eff, args.normalize_method, args.time_limit_sec,
                                                 not args.no_enforce_tree, args.mode, args.progen_matrix)
                    print(f"  >>> best k={best_k}, Jaccard={jd:.6f}")
                    results.append((typ, cells, idd, jd, best_k))
                except Exception as e:
                    print(f"  ERROR id={idd}: {e}")
                    results.append((typ, cells, idd, None, None))

    # summary
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    print("\n===== Summary =====")
    print(f"{'Type':<8} {'Cells':<9} {'ID':<6} {'Jaccard':<10} {'Best-k':<6}")
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["type","cells","id","jaccard","best_k"])
        for typ, cells, idd, jd, bk in results:
            jd_s = "ERROR" if jd is None else f"{jd:.6f}"
            bk_s = "" if bk is None else str(bk)
            print(f"{typ:<8} {cells:<9} {idd:<6} {jd_s:<10} {bk_s:<6}")
            w.writerow([typ, cells, idd, "" if jd is None else f"{jd:.6f}", "" if bk is None else bk])

    print(f"\nSaved {args.csv}")

if __name__ == "__main__":
    main()
