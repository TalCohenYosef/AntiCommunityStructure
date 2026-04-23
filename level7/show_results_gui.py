import json
import os
import argparse
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_color(score):
    if score >= 0.9:
        return "#16a34a"
    elif score >= 0.7:
        return "#22c55e"
    elif score >= 0.55:
        return "#eab308"
    else:
        return "#ef4444"


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

    # Plot 2: Score
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    ax2.barh([0], [score], height=0.5, color=score_color(score))
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Score S(G,{k}) = {score:.4f}")
    ax2.set_xlabel("Score")
    ax2.set_yticks([])
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/score_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: Loss curve
    if loss_history:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(range(1, len(loss_history) + 1), loss_history)
        ax3.set_title("Loss vs Epoch")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(f"{output_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

    print(f"\nPlots saved to: {output_dir}/")
    print("  cluster_sizes.png")
    print("  score_visualization.png")
    if loss_history:
        print("  loss_curve.png")
    print("=" * 80)


def build_gui(results):
    assignments = results["assignments"]
    score = results["score"]
    intra_weight = results["intra_cluster_weight"]
    total_weight = results["total_edge_weight"]
    loss_history = results.get("loss_history", [])

    k = get_k_from_results(results)
    cluster_sizes = get_cluster_sizes(assignments, k)
    labels = [f"Cluster {i}" for i in range(k)]

    num_nodes = len(assignments)


    root = tk.Tk()
    root.title(f"Anti-Community GNN Results (k={k})")
    root.geometry("1280x900")
    root.configure(bg="#0f172a")

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Title.TLabel", background="#0f172a", foreground="white", font=("Segoe UI", 22, "bold"))
    style.configure("Subtitle.TLabel", background="#0f172a", foreground="#cbd5e1", font=("Segoe UI", 11))
    style.configure("Card.TFrame", background="#1e293b")
    style.configure("CardTitle.TLabel", background="#1e293b", foreground="#93c5fd", font=("Segoe UI", 13, "bold"))
    style.configure("CardValue.TLabel", background="#1e293b", foreground="white", font=("Segoe UI", 18, "bold"))
    style.configure("Text.TLabel", background="#1e293b", foreground="#e2e8f0", font=("Segoe UI", 11))
    style.configure("Section.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 15, "bold"))

    main = ttk.Frame(root, padding=20)
    main.pack(fill="both", expand=True)

    header = ttk.Frame(main)
    header.pack(fill="x", pady=(0, 16))

    ttk.Label(
        header,
        text=f"FINAL SUMMARY - ANTI COMMUNITY GNN (k={k})",
        style="Title.TLabel"
    ).pack(anchor="w")
    ttk.Label(
        header,
        text="Visual summary of the current experiment",
        style="Subtitle.TLabel"
    ).pack(anchor="w", pady=(4, 0))

    cards = ttk.Frame(main)
    cards.pack(fill="x", pady=(0, 18))

    def create_card(parent, title, value, row, col):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel").pack(anchor="w", pady=(10, 0))

    for i in range(3):
        cards.columnconfigure(i, weight=1)

    create_card(cards, "Nodes", f"{num_nodes}", 0, 0)
    create_card(cards, f"Score S(G,{k})", f"{score:.4f}", 0, 1)
    create_card(cards, "Clusters", f"{k}", 0, 2)

    create_card(cards, "Total Edge Weight", f"{total_weight:.0f}", 1, 0)

    create_card(cards, "Intra-Cluster Weight", f"{intra_weight:.0f}", 1, 2)

    middle = ttk.Frame(main)
    middle.pack(fill="both", expand=True, pady=(0, 18))

    left_panel = ttk.Frame(middle, style="Card.TFrame", padding=16)
    left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

    right_panel = ttk.Frame(middle, style="Card.TFrame", padding=16)
    right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))

    ttk.Label(left_panel, text="Cluster Sizes", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    ttk.Label(
        left_panel,
        text="This plot shows the number of nodes in each cluster.",
        style="Text.TLabel",
        wraplength=500,
        justify="left"
    ).pack(anchor="w", pady=(0, 10))

    fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))
    bars = ax1.bar(labels, cluster_sizes)
    ax1.set_title("Cluster Sizes")
    ax1.set_ylabel("Number of Nodes")
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

    canvas1 = FigureCanvasTkAgg(fig1, master=left_panel)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, pady=(5, 10))

    ttk.Label(right_panel, text="Training Loss", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    ttk.Label(
        right_panel,
        text="This plot shows the loss during training.",
        style="Text.TLabel",
        wraplength=500,
        justify="left"
    ).pack(anchor="w", pady=(0, 10))

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    if loss_history:
        ax2.plot(range(1, len(loss_history) + 1), loss_history)
        ax2.set_title("Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No loss history available", ha="center", va="center")
        ax2.set_axis_off()
    fig2.tight_layout()

    canvas2 = FigureCanvasTkAgg(fig2, master=right_panel)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, pady=(5, 10))

    score_frame = ttk.Frame(main, style="Card.TFrame", padding=16)
    score_frame.pack(fill="x", pady=(0, 18))

    ttk.Label(score_frame, text="Score", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

    score_canvas = tk.Canvas(score_frame, width=1000, height=50, bg="#1e293b", highlightthickness=0)
    score_canvas.pack(fill="x")

    score_canvas.create_rectangle(30, 18, 970, 32, fill="#334155", outline="")
    fill_width = 30 + (940 * score)
    score_canvas.create_rectangle(30, 18, fill_width, 32, fill=score_color(score), outline="")
    score_canvas.create_text(30, 8, text="0.0", anchor="w", fill="white", font=("Segoe UI", 10))
    score_canvas.create_text(970, 8, text="1.0", anchor="e", fill="white", font=("Segoe UI", 10))
    score_canvas.create_text(
        min(max(fill_width, 60), 940),
        42,
        text=f"S(G,{k}) = {score:.4f}",
        fill="white",
        font=("Segoe UI", 11, "bold")
    )

    bottom = ttk.Frame(main, style="Card.TFrame", padding=18)
    bottom.pack(fill="both", expand=True)

    ttk.Label(bottom, text="Cluster Sizes", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

    cluster_text = "\n".join([f"Cluster {i}: {size}" for i, size in enumerate(cluster_sizes)])
    ttk.Label(
        bottom,
        text=cluster_text,
        style="Text.TLabel",
        wraplength=1080,
        justify="left"
    ).pack(anchor="w")

    root.mainloop()


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
        help="Directory to save plots in headless mode"
    )

    args = parser.parse_args()

    results = load_results(args.results_path)

    has_display = bool(os.environ.get("DISPLAY"))

    if has_display:
        print("Display detected - launching GUI mode...")
        build_gui(results)
    else:
        print("No display detected - running in headless mode...")
        save_plots_and_print_results(results, output_dir=args.output_dir)