import json
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_k_from_results(results):
    if "k" in results:
        return results["k"]

    assignments = results.get("assignments", [])
    if not assignments:
        return 0

    max_cluster = max(item["cluster"] for item in assignments)
    return max_cluster + 1


def get_cluster_sizes(assignments, k):
    sizes = [0] * k
    for item in assignments:
        c = item["cluster"]
        if 0 <= c < k:
            sizes[c] += 1
    return sizes


def save_plots_and_print_results(results, output_dir="level7"):
    os.makedirs(output_dir, exist_ok=True)

    assignments = results["assignments"]
    score = results["score"]
    intra_weight = results["intra_cluster_weight"]
    total_weight = results["total_edge_weight"]
    loss_history = results.get("loss_history", [])

    k = get_k_from_results(results)
    cluster_sizes = get_cluster_sizes(assignments, k)
    labels = [f"Cluster {i}" for i in range(k)]

    num_nodes = len(assignments)

    print("\n" + "=" * 80)
    print(f"ANTI-COMMUNITY GNN RESULTS (k={k})")
    print("=" * 80)
    print(f"Total Nodes: {num_nodes}")
    print(f"Score S(G,{k}): {score:.4f}")
    print(f"Total Edge Weight: {total_weight:.0f}")
    print(f"Intra-Cluster Weight: {intra_weight:.0f}")

    print("\nCluster Sizes:")
    for i, size in enumerate(cluster_sizes):
        print(f"Cluster {i}: {size}")

    # Plot 1: Cluster sizes
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(labels, cluster_sizes)
    ax1.set_title("Cluster Sizes")
    ax1.set_ylabel("Number of Nodes")
    ax1.set_xlabel("Clusters")
    ax1.grid(True, alpha=0.3)

    ymax = max(cluster_sizes) if cluster_sizes else 1
    for bar, val in zip(bars, cluster_sizes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.01,
            str(val),
            ha="center",
            va="bottom"
        )

    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Loss curve
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    if loss_history:
        epochs = list(range(1, len(loss_history) + 1))
        ax2.plot(epochs, loss_history, marker="o", color="#38bdf8")
        ax2.set_title("Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No loss history available", ha="center", va="center", fontsize=14)
        ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: Score bar visualization
    if score >= 0.9:
        bar_color = "#16a34a"
        grade = "Excellent (≥90%)"
    elif score >= 0.7:
        bar_color = "#22c55e"
        grade = "Good (≥70%)"
    elif score >= 0.55:
        bar_color = "#eab308"
        grade = "Fair (≥55%)"
    else:
        bar_color = "#ef4444"
        grade = "Poor (<55%)"

    fig3, ax3 = plt.subplots(figsize=(10, 2))
    ax3.barh([0], [score], color=bar_color, height=0.6)
    ax3.set_xlim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Score (0.0 = No Structure, 1.0 = Perfect Bipartite)")
    ax3.set_title(f"Anti-Community Score: {score:.4f}", fontweight="bold", fontsize=14)
    ax3.text(1.01, 0, grade, va="center", ha="left", fontsize=11, fontweight="bold", transform=ax3.get_yaxis_transform())
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/score_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"\nPlots saved to: {output_dir}/")
    print("  cluster_sizes.png")
    print("  score_visualization.png")
    print("  loss_curve.png")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show Anti-Community GNN Results")
    parser.add_argument(
        "--results-path",
        type=str,
        default="level6/k2_results.json",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="level7",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    results = load_results(args.results_path)
    save_plots_and_print_results(results, output_dir=args.output_dir)
