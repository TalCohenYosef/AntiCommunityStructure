import json
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def interpret_score(score):
    if score >= 0.9:
        return "The graph is very close to a perfect 2-partite structure."
    elif score >= 0.7:
        return "The graph shows a strong bipartite / anti-community tendency."
    elif score >= 0.55:
        return "The graph shows a moderate bipartite / anti-community tendency."
    else:
        return "The graph does not show a strong bipartite / anti-community structure."


def interpret_balance(balance_ratio):
    if balance_ratio < 0.2:
        return "The partition is highly unbalanced."
    elif balance_ratio < 0.5:
        return "The partition is somewhat unbalanced."
    else:
        return "The partition is reasonably balanced."


def final_conclusion(score, balance_ratio):
    if score >= 0.55 and balance_ratio >= 0.5:
        return (
            "The model found a meaningful and reasonably balanced "
            "2-way anti-community partition."
        )
    elif score >= 0.55 and balance_ratio < 0.5:
        return (
            "The model found a moderate 2-way anti-community structure, "
            "but the resulting partition is quite unbalanced."
        )
    else:
        return (
            "The current k=2 model does not yet provide a strong and balanced "
            "anti-community partition."
        )


def score_color(score):
    if score >= 0.9:
        return "#16a34a"
    elif score >= 0.7:
        return "#22c55e"
    elif score >= 0.55:
        return "#eab308"
    else:
        return "#ef4444"


def build_gui(results):
    score = results["score"]
    intra_weight = results["intra_cluster_weight"]
    total_weight = results["total_edge_weight"]
    assignments = results["assignments"]

    cluster_0 = sum(1 for item in assignments if item["cluster"] == 0)
    cluster_1 = sum(1 for item in assignments if item["cluster"] == 1)
    num_nodes = len(assignments)

    inter_weight = total_weight - intra_weight
    balance_ratio = (
        min(cluster_0, cluster_1) / max(cluster_0, cluster_1)
        if max(cluster_0, cluster_1) > 0
        else 0.0
    )

    root = tk.Tk()
    root.title("Anti-Community GNN Results (k=2)")
    root.geometry("1200x800")
    root.configure(bg="#0f172a")

    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "Title.TLabel",
        background="#0f172a",
        foreground="white",
        font=("Segoe UI", 22, "bold")
    )
    style.configure(
        "Subtitle.TLabel",
        background="#0f172a",
        foreground="#cbd5e1",
        font=("Segoe UI", 11)
    )
    style.configure(
        "Card.TFrame",
        background="#1e293b"
    )
    style.configure(
        "CardTitle.TLabel",
        background="#1e293b",
        foreground="#93c5fd",
        font=("Segoe UI", 13, "bold")
    )
    style.configure(
        "CardValue.TLabel",
        background="#1e293b",
        foreground="white",
        font=("Segoe UI", 18, "bold")
    )
    style.configure(
        "Text.TLabel",
        background="#1e293b",
        foreground="#e2e8f0",
        font=("Segoe UI", 11)
    )
    style.configure(
        "Section.TLabel",
        background="#0f172a",
        foreground="#f8fafc",
        font=("Segoe UI", 15, "bold")
    )

    main = ttk.Frame(root, padding=20)
    main.pack(fill="both", expand=True)

    header = ttk.Frame(main)
    header.pack(fill="x", pady=(0, 16))

    ttk.Label(
        header,
        text="FINAL SUMMARY - ANTI COMMUNITY GNN (k=2)",
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
        return card

    for i in range(3):
        cards.columnconfigure(i, weight=1)

    create_card(cards, "Nodes", f"{num_nodes}", 0, 0)
    create_card(cards, "Score S(G,2)", f"{score:.4f}", 0, 1)
    create_card(cards, "Balance Ratio", f"{balance_ratio:.4f}", 0, 2)

    create_card(cards, "Total Edge Weight", f"{total_weight:.0f}", 1, 0)
    create_card(cards, "Inter-Cluster Weight", f"{inter_weight:.0f}", 1, 1)
    create_card(cards, "Intra-Cluster Weight", f"{intra_weight:.0f}", 1, 2)

    middle = ttk.Frame(main)
    middle.pack(fill="both", expand=True, pady=(0, 18))

    left_panel = ttk.Frame(middle, style="Card.TFrame", padding=16)
    left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

    right_panel = ttk.Frame(middle, style="Card.TFrame", padding=16)
    right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))

    ttk.Label(left_panel, text="Cluster Separation View", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    ttk.Label(
        left_panel,
        text="This view shows how the nodes are distributed between the two clusters.",
        style="Text.TLabel",
        wraplength=500,
        justify="left"
    ).pack(anchor="w", pady=(0, 10))

    fig1, ax1 = plt.subplots(figsize=(5, 3.2))
    counts = [cluster_0, cluster_1]
    labels = ["Cluster 0", "Cluster 1"]
    bars = ax1.bar(labels, counts)
    ax1.set_title("Cluster Sizes")
    ax1.set_ylabel("Number of Nodes")
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, str(val),
                 ha="center", va="bottom")
    fig1.tight_layout()

    canvas1 = FigureCanvasTkAgg(fig1, master=left_panel)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, pady=(5, 10))

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
    ax2.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.set_title("Partition Proportions")
    fig2.tight_layout()

    canvas2 = FigureCanvasTkAgg(fig2, master=right_panel)
    ttk.Label(right_panel, text="Partition Proportions", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, pady=(5, 10))

    score_frame = ttk.Frame(main, style="Card.TFrame", padding=16)
    score_frame.pack(fill="x", pady=(0, 18))

    ttk.Label(score_frame, text="Score Quality", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

    score_canvas = tk.Canvas(score_frame, width=1000, height=50, bg="#1e293b", highlightthickness=0)
    score_canvas.pack(fill="x")

    score_canvas.create_rectangle(30, 18, 970, 32, fill="#334155", outline="")
    fill_width = 30 + (940 * score)
    score_canvas.create_rectangle(30, 18, fill_width, 32, fill=score_color(score), outline="")
    score_canvas.create_text(30, 8, text="0.0", anchor="w", fill="white", font=("Segoe UI", 10))
    score_canvas.create_text(970, 8, text="1.0", anchor="e", fill="white", font=("Segoe UI", 10))
    score_canvas.create_text(fill_width, 42, text=f"S(G,2) = {score:.4f}", fill="white", font=("Segoe UI", 11, "bold"))

    bottom = ttk.Frame(main, style="Card.TFrame", padding=18)
    bottom.pack(fill="both", expand=True)

    ttk.Label(bottom, text="Interpretation", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    ttk.Label(
        bottom,
        text=interpret_score(score),
        style="Text.TLabel",
        wraplength=1080,
        justify="left"
    ).pack(anchor="w", pady=(0, 8))

    ttk.Label(
        bottom,
        text=interpret_balance(balance_ratio),
        style="Text.TLabel",
        wraplength=1080,
        justify="left"
    ).pack(anchor="w", pady=(0, 8))

    ttk.Label(bottom, text="Conclusion", style="Section.TLabel").pack(anchor="w", pady=(15, 10))
    ttk.Label(
        bottom,
        text=final_conclusion(score, balance_ratio),
        style="Text.TLabel",
        wraplength=1080,
        justify="left"
    ).pack(anchor="w")

    root.mainloop()


if __name__ == "__main__":
    results_path = "level6/k2_results.json"
    results = load_results(results_path)
    build_gui(results)