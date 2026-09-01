
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.4,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


COLORS = {
    "cosine": "#1f77b4",
    "js": "#2ca02c",
    "dominant": "#ff7f0e",
    "support": "#9467bd",
    "damage": "#d62728",
    "isolated": "#7f7f7f",
}


def load_dataset_from_repo(dataset_name: str, root_dir: str):
    try:
        from dataset import get_dataset
    except Exception as e:
        raise ImportError(
            "Could not import get_dataset from dataset.py. Run this script from "
            "the graphswav-main repo root, or make sure dataset.py is on PYTHONPATH."
        ) from e
    data = get_dataset(dataset_name, root_dir=root_dir)
    if isinstance(data, list):
        data = data[0]
    return data


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def remove_self_loops_np(edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    mask = src != dst
    return edge_index[:, mask]


def to_unique_undirected_pairs(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = remove_self_loops_np(edge_index)
    src, dst = edge_index
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    pairs = np.stack([u, v], axis=1)
    valid = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] < num_nodes) & (pairs[:, 1] < num_nodes)
    valid &= pairs[:, 0] != pairs[:, 1]
    pairs = pairs[valid]
    if pairs.shape[0] == 0:
        return pairs.astype(np.int64)
    return np.unique(pairs.astype(np.int64), axis=0)


def directed_edges_from_pairs(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pairs.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
    return src, dst


def unique_directed_edges(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    edge_index = remove_self_loops_np(edge_index)
    src, dst = edge_index
    valid = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes) & (src != dst)
    edges = np.stack([src[valid], dst[valid]], axis=1).astype(np.int64)
    if edges.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    edges = np.unique(edges, axis=0)
    return edges[:, 0], edges[:, 1]


@dataclass
class Fingerprint:
    dist: np.ndarray
    counts: np.ndarray
    degree: np.ndarray
    dominant: np.ndarray
    support: np.ndarray


def compute_label_fingerprint(num_nodes: int, labels: np.ndarray, num_classes: int, src: np.ndarray, dst: np.ndarray) -> Fingerprint:
    counts = np.zeros((num_nodes, num_classes), dtype=np.float64)
    if src.size > 0:
        neighbour_labels = labels[dst].astype(np.int64)
        np.add.at(counts, (src.astype(np.int64), neighbour_labels), 1.0)
    degree = counts.sum(axis=1)
    dist = np.zeros_like(counts, dtype=np.float64)
    nonzero = degree > 0
    dist[nonzero] = counts[nonzero] / degree[nonzero, None]
    dominant = np.full(num_nodes, -1, dtype=np.int64)
    dominant[nonzero] = np.argmax(dist[nonzero], axis=1)
    support = counts > 0
    return Fingerprint(dist=dist, counts=counts, degree=degree, dominant=dominant, support=support)


def normalized_js_similarity(p: np.ndarray, q: np.ndarray, valid: np.ndarray) -> float:
    eps = 1e-12
    idx = np.where(valid)[0]
    if idx.size == 0:
        return float("nan")
    p_sub = p[idx]
    q_sub = q[idx]
    q_nonzero = q_sub.sum(axis=1) > 0
    sims = np.zeros(idx.size, dtype=np.float64)
    if q_nonzero.any():
        pp = p_sub[q_nonzero]
        qq = q_sub[q_nonzero]
        m = 0.5 * (pp + qq)
        kl_pm = np.sum(pp * (np.log(pp + eps) - np.log(m + eps)), axis=1)
        kl_qm = np.sum(qq * (np.log(qq + eps) - np.log(m + eps)), axis=1)
        jsd = 0.5 * (kl_pm + kl_qm)
        sims[q_nonzero] = 1.0 - jsd / math.log(2.0)
        sims = np.clip(sims, 0.0, 1.0)
    return float(np.mean(sims))


def cosine_similarity_mean(p: np.ndarray, q: np.ndarray, valid: np.ndarray) -> float:
    idx = np.where(valid)[0]
    if idx.size == 0:
        return float("nan")
    pp = p[idx]
    qq = q[idx]
    dot = np.sum(pp * qq, axis=1)
    pn = np.linalg.norm(pp, axis=1)
    qn = np.linalg.norm(qq, axis=1)
    denom = pn * qn
    sims = np.zeros(idx.size, dtype=np.float64)
    nz = denom > 0
    sims[nz] = dot[nz] / denom[nz]
    return float(np.mean(np.clip(sims, 0.0, 1.0)))


def total_variation_distance_mean(p: np.ndarray, q: np.ndarray, valid: np.ndarray) -> float:
    idx = np.where(valid)[0]
    if idx.size == 0:
        return float("nan")
    tv = 0.5 * np.sum(np.abs(p[idx] - q[idx]), axis=1)
    return float(np.mean(tv))


def support_retention_mean(orig_support: np.ndarray, corr_support: np.ndarray, valid: np.ndarray) -> float:
    idx = np.where(valid)[0]
    if idx.size == 0:
        return float("nan")
    orig = orig_support[idx]
    corr = corr_support[idx]
    denom = orig.sum(axis=1)
    keep = np.logical_and(orig, corr).sum(axis=1)
    nonzero = denom > 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(keep[nonzero] / denom[nonzero]))


def dominant_transition_matrix(orig_dom: np.ndarray, corr_dom: np.ndarray, valid: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes + 1), dtype=np.float64)
    idx = np.where(valid)[0]
    for i in idx:
        a = orig_dom[i]
        b = corr_dom[i]
        if a < 0:
            continue
        col = num_classes if b < 0 else b
        mat[a, col] += 1.0
    row_sum = mat.sum(axis=1)
    nz = row_sum > 0
    mat[nz] = mat[nz] / row_sum[nz, None]
    return mat


def compute_metrics(orig: Fingerprint, corr: Fingerprint) -> Dict[str, float]:
    valid = orig.degree > 0
    changed = np.zeros_like(valid, dtype=bool)
    changed[valid] = corr.dominant[valid] != orig.dominant[valid]
    isolated_after = valid & (corr.degree == 0)
    return {
        "num_valid_nodes": int(valid.sum()),
        "mean_degree_original": float(orig.degree[valid].mean()) if valid.any() else float("nan"),
        "mean_degree_corrupted": float(corr.degree[valid].mean()) if valid.any() else float("nan"),
        "degree_retention": float(corr.degree[valid].sum() / max(orig.degree[valid].sum(), 1.0)) if valid.any() else float("nan"),
        "fingerprint_cosine_retention": cosine_similarity_mean(orig.dist, corr.dist, valid),
        "js_similarity": normalized_js_similarity(orig.dist, corr.dist, valid),
        "total_variation_distance": total_variation_distance_mean(orig.dist, corr.dist, valid),
        "support_retention": support_retention_mean(orig.support, corr.support, valid),
        "dominant_relation_retention": float(1.0 - changed[valid].mean()) if valid.any() else float("nan"),
        "dominant_relation_changed": float(changed[valid].mean()) if valid.any() else float("nan"),
        "isolated_after_corruption": float(isolated_after[valid].mean()) if valid.any() else float("nan"),
    }


def corrupt_edges(rng: np.random.Generator, drop_prob: float, undirected_pairs: Optional[np.ndarray] = None,
                  directed_src: Optional[np.ndarray] = None, directed_dst: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    p = float(drop_prob)
    if undirected_pairs is not None:
        m = undirected_pairs.shape[0]
        if p <= 0:
            kept = undirected_pairs
        elif p >= 1:
            kept = undirected_pairs[:0]
        else:
            kept = undirected_pairs[rng.random(m) >= p]
        src, dst = directed_edges_from_pairs(kept)
        return src, dst, int(kept.shape[0])
    assert directed_src is not None and directed_dst is not None
    m = directed_src.size
    if p <= 0:
        src, dst = directed_src, directed_dst
    elif p >= 1:
        src, dst = directed_src[:0], directed_dst[:0]
    else:
        mask = rng.random(m) >= p
        src, dst = directed_src[mask], directed_dst[mask]
    return src, dst, int(src.size)


def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    skip = {"dataset", "drop_prob", "trial", "seed", "retained_base_edges", "num_base_edges"}
    metric_cols = [c for c in df.columns if c not in skip]
    rows = []
    for p, g in df.groupby("drop_prob"):
        row = {"drop_prob": float(p), "num_trials": int(len(g))}
        for c in metric_cols:
            vals = g[c].to_numpy(dtype=float)
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c}_mean"] = mean
            row[f"{c}_std"] = std
            row[f"{c}_ci95"] = float(1.96 * std / math.sqrt(max(len(vals), 1)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("drop_prob")


def plot_with_ci(ax, summary: pd.DataFrame, y_mean: str, y_ci: str, label: str, color: str, marker: str):
    xs = summary["drop_prob"].to_numpy(dtype=float)
    ys = summary[y_mean].to_numpy(dtype=float)
    cis = summary[y_ci].to_numpy(dtype=float)
    ax.plot(xs, ys, marker=marker, color=color, label=label)
    ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.14, linewidth=0)


def savefig(fig, out_dir: str, basename: str) -> None:
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{basename}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


def make_preservation_plot(summary: pd.DataFrame, out_dir: str, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plot_with_ci(ax, summary, "fingerprint_cosine_retention_mean", "fingerprint_cosine_retention_ci95", "Fingerprint cosine retention", COLORS["cosine"], "o")
    plot_with_ci(ax, summary, "js_similarity_mean", "js_similarity_ci95", "1 - normalized JSD", COLORS["js"], "s")
    plot_with_ci(ax, summary, "dominant_relation_retention_mean", "dominant_relation_retention_ci95", "Dominant relation retained", COLORS["dominant"], "^")
    plot_with_ci(ax, summary, "support_retention_mean", "support_retention_ci95", "Neighbour-label support retained", COLORS["support"], "D")
    ax.set_title("Relational fingerprints degrade under random edge deletion")
    ax.set_xlabel("Edge-drop probability")
    ax.set_ylabel("Preservation score")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", frameon=True)
    ax.text(0.99, 0.03, dataset, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.35")
    savefig(fig, out_dir, "semantic_relation_preservation")


def make_damage_plot(summary: pd.DataFrame, out_dir: str, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_with_ci(ax, summary, "dominant_relation_changed_mean", "dominant_relation_changed_ci95", "Dominant relation changed", COLORS["damage"], "o")
    plot_with_ci(ax, summary, "isolated_after_corruption_mean", "isolated_after_corruption_ci95", "Node becomes isolated", COLORS["isolated"], "s")
    plot_with_ci(ax, summary, "total_variation_distance_mean", "total_variation_distance_ci95", "Total variation shift", COLORS["dominant"], "^")
    ax.set_title("Random edge deletion changes semantic neighbourhoods")
    ax.set_xlabel("Edge-drop probability")
    ax.set_ylabel("Damage score")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.99, 0.03, dataset, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.35")
    savefig(fig, out_dir, "semantic_relation_damage")


def make_transition_heatmap(mat: np.ndarray, out_dir: str, p: float, class_names: Optional[List[str]] = None) -> None:
    num_classes = mat.shape[0]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.42 * (num_classes + 1)), max(5.5, 0.35 * num_classes)))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma_r")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of nodes")
    labels_y = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    labels_x = labels_y + ["isolated"]
    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Dominant neighbour label after corruption")
    ax.set_ylabel("Original dominant neighbour label")
    ax.set_title(f"Dominant relation transitions under edge dropout p={p:g}")
    for i in range(num_classes):
        for j in range(num_classes + 1):
            val = mat[i, j]
            if val >= 0.15:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    safe_p = str(p).replace(".", "p")
    savefig(fig, out_dir, f"dominant_relation_transition_p{safe_p}")


def make_degree_bin_plot(orig: Fingerprint, corr_by_trial: List[Fingerprint], out_dir: str, p: float) -> None:
    deg = orig.degree
    valid = deg > 0
    bins = np.array([1, 2, 3, 5, 10, 20, 50, np.inf], dtype=float)
    labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
    rows = []
    for trial_id, corr in enumerate(corr_by_trial):
        changed = np.zeros_like(valid, dtype=bool)
        changed[valid] = corr.dominant[valid] != orig.dominant[valid]
        for b, name in enumerate(labels):
            lo, hi = bins[b], bins[b + 1]
            if np.isinf(hi):
                m = valid & (deg >= lo)
            elif hi == lo + 1:
                m = valid & (deg == lo)
            else:
                m = valid & (deg >= lo) & (deg < hi)
            if m.sum() > 0:
                rows.append({"trial": trial_id, "degree_bin": name, "num_nodes": int(m.sum()), "failure_rate": float(changed[m].mean())})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    s = df.groupby("degree_bin", sort=False).agg(failure_mean=("failure_rate", "mean"), failure_std=("failure_rate", "std"), num_nodes=("num_nodes", "mean")).reindex(labels).reset_index()
    s["failure_ci95"] = 1.96 * s["failure_std"].fillna(0.0) / math.sqrt(max(len(corr_by_trial), 1))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(s))
    ys = s["failure_mean"].to_numpy(dtype=float)
    cis = s["failure_ci95"].to_numpy(dtype=float)
    ax.bar(xs, ys, yerr=cis, capsize=3, color=COLORS["damage"], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(s["degree_bin"].tolist())
    ax.set_ylim(0.0, min(1.0, max(0.15, float(np.nanmax(ys + cis)) + 0.1)))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Original node degree")
    ax.set_ylabel("Dominant relation changed")
    ax.set_title(f"Low-degree nodes are most fragile under edge dropout p={p:g}")
    for x, y, n in zip(xs, ys, s["num_nodes"].to_numpy(dtype=float)):
        if not np.isnan(y):
            ax.text(x, y + 0.02, f"n≈{int(round(n))}", ha="center", va="bottom", fontsize=8, rotation=90)
    safe_p = str(p).replace(".", "p")
    savefig(fig, out_dir, f"degree_bin_damage_p{safe_p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-only audit of semantic relation loss under stochastic edge corruption.")
    parser.add_argument("--dataset", type=str, default="roman-empire")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="relation_corruption_results/roman_empire")
    parser.add_argument("--drop_probs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--make_undirected", action="store_true", help="Use unique undirected edges and drop them symmetrically. Recommended.")
    parser.add_argument("--directed", action="store_true", help="Use directed edges as stored. Overrides --make_undirected.")
    parser.add_argument("--plot_transition_p", type=float, default=0.3, help="Drop probability for transition heatmap. Use -1 to disable.")
    parser.add_argument("--plot_degree_p", type=float, default=0.3, help="Drop probability for degree-bin plot. Use -1 to disable.")
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_publication_style()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[*] Loading dataset: {args.dataset}")
    data = load_dataset_from_repo(args.dataset, args.root_dir)
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Dataset has no edge_index.")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Dataset has no labels y. This audit needs labels as an analysis signal.")
    edge_index = tensor_to_numpy(data.edge_index).astype(np.int64)
    labels = tensor_to_numpy(data.y).astype(np.int64).reshape(-1)
    num_nodes = int(data.num_nodes) if getattr(data, "num_nodes", None) is not None else int(labels.shape[0])
    num_classes = int(labels.max() + 1)
    if labels.min() < 0:
        raise ValueError("Labels must be nonnegative integer class IDs.")
    use_undirected = args.make_undirected and not args.directed
    if use_undirected:
        pairs = to_unique_undirected_pairs(edge_index, num_nodes)
        base_src, base_dst = directed_edges_from_pairs(pairs)
        num_base_edges = int(pairs.shape[0])
        graph_mode = "unique_undirected_pairs"
    else:
        pairs = None
        base_src, base_dst = unique_directed_edges(edge_index, num_nodes)
        num_base_edges = int(base_src.size)
        graph_mode = "unique_directed_edges"
    print(f"[*] Nodes: {num_nodes}")
    print(f"[*] Classes: {num_classes}")
    print(f"[*] Graph mode: {graph_mode}")
    print(f"[*] Base edges for corruption: {num_base_edges}")
    orig = compute_label_fingerprint(num_nodes, labels, num_classes, base_src, base_dst)
    valid_nodes = int((orig.degree > 0).sum())
    print(f"[*] Nodes with at least one neighbour: {valid_nodes}/{num_nodes}")
    print(f"[*] Mean original degree over valid nodes: {orig.degree[orig.degree > 0].mean():.3f}")
    rng_master = np.random.default_rng(args.seed)
    rows = []
    transition_corrs: List[Fingerprint] = []
    degree_corrs: List[Fingerprint] = []
    transition_p = None if args.plot_transition_p < 0 else float(args.plot_transition_p)
    degree_p = None if args.plot_degree_p < 0 else float(args.plot_degree_p)
    for p in args.drop_probs:
        p = float(p)
        print(f"[*] Auditing edge-drop p={p:g}")
        for trial in range(args.num_trials):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            corr_src, corr_dst, retained = corrupt_edges(
                rng=rng,
                drop_prob=p,
                undirected_pairs=pairs,
                directed_src=None if use_undirected else base_src,
                directed_dst=None if use_undirected else base_dst,
            )
            corr = compute_label_fingerprint(num_nodes, labels, num_classes, corr_src, corr_dst)
            m = compute_metrics(orig, corr)
            m.update({"dataset": args.dataset, "drop_prob": p, "trial": trial, "seed": seed, "num_base_edges": num_base_edges, "retained_base_edges": retained})
            rows.append(m)
            if transition_p is not None and abs(p - transition_p) < 1e-12:
                transition_corrs.append(corr)
            if degree_p is not None and abs(p - degree_p) < 1e-12:
                degree_corrs.append(corr)
    per_trial = pd.DataFrame(rows)
    summary = summarize_trials(per_trial)
    per_trial_path = os.path.join(args.out_dir, "relation_corruption_per_trial.csv")
    summary_path = os.path.join(args.out_dir, "relation_corruption_summary.csv")
    per_trial.to_csv(per_trial_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {per_trial_path}")
    print(f"[saved] {summary_path}")
    meta = {
        "dataset": args.dataset,
        "root_dir": args.root_dir,
        "num_nodes": num_nodes,
        "num_classes": num_classes,
        "graph_mode": graph_mode,
        "num_base_edges": num_base_edges,
        "drop_probs": [float(x) for x in args.drop_probs],
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    make_preservation_plot(summary, args.out_dir, args.dataset)
    make_damage_plot(summary, args.out_dir, args.dataset)
    if transition_p is not None and len(transition_corrs) > 0:
        mat = dominant_transition_matrix(orig.dominant, transition_corrs[0].dominant, orig.degree > 0, num_classes)
        np.save(os.path.join(args.out_dir, f"dominant_relation_transition_p{str(transition_p).replace('.', 'p')}.npy"), mat)
        make_transition_heatmap(mat, args.out_dir, transition_p)
    elif transition_p is not None:
        print(f"[warn] plot_transition_p={transition_p} was not in --drop_probs; skipping transition heatmap.")
    if degree_p is not None and len(degree_corrs) > 0:
        make_degree_bin_plot(orig, degree_corrs, args.out_dir, degree_p)
    elif degree_p is not None:
        print(f"[warn] plot_degree_p={degree_p} was not in --drop_probs; skipping degree-bin plot.")
    if args.save_npz:
        np.savez_compressed(os.path.join(args.out_dir, "original_relational_fingerprints.npz"), dist=orig.dist, counts=orig.counts, degree=orig.degree, dominant=orig.dominant, support=orig.support.astype(np.uint8), labels=labels)
    print("\nDone. Key files:")
    print(f"  {summary_path}")
    print(f"  {os.path.join(args.out_dir, 'semantic_relation_preservation.png')}")
    print(f"  {os.path.join(args.out_dir, 'semantic_relation_damage.png')}")


if __name__ == "__main__":
    main()