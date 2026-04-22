import os
import sys
import torch
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from level3.load_gnn_data import load_gnn_input
from level4.model import AntiCommunityGNN


def compute_hard_score(node_ids, assignments, edge_index, edge_weight):
    """
    Compute the hard anti-community score S(G,2)
    using hard assignments from argmax.
    """
    intra_weight = 0.0
    total_weight = 0.0

    src = edge_index[0]
    dst = edge_index[1]

    # Count each undirected edge only once
    for i in range(edge_index.shape[1]):
        u = src[i].item()
        v = dst[i].item()

        if u < v:
            w = edge_weight[i].item()
            total_weight += w

            if assignments[u].item() == assignments[v].item():
                intra_weight += w

    score = 1.0 - (intra_weight / total_weight) if total_weight > 0 else 0.0
    return score, intra_weight, total_weight


def soft_anticommunity_loss(p, edge_index, edge_weight):
    """
    Loss = sum over edges of:
           edge_weight * dot(p_u, p_v)

    Only neighboring nodes contribute to the loss.
    Each undirected edge is counted once.
    """
    src = edge_index[0]
    dst = edge_index[1]

    mask = src < dst
    src = src[mask]
    dst = dst[mask]
    edge_weight = edge_weight[mask]

    dot_products = (p[src] * p[dst]).sum(dim=1)
    loss = (edge_weight * dot_products).sum()
    return loss


if __name__ == "__main__":
    data, node_ids = load_gnn_input("level2/gnn_input.json")

    model = AntiCommunityGNN(in_channels=2, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 10000
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        p = model(data.x, data.edge_index, data.edge_weight)
        loss = soft_anticommunity_loss(p, data.edge_index, data.edge_weight)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        #print(f"Epoch {epoch:03d} | Loss = {loss_value:.6f}")

    print("\nTraining finished.")

    # Plot loss curve
    os.makedirs("level6", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("level6/loss_curve.png")
    plt.show()

    model.eval()
    with torch.no_grad():
        p = model(data.x, data.edge_index, data.edge_weight)
        assignments = torch.argmax(p, dim=1)
        hard_one_hot = F.one_hot(assignments, num_classes=2).float()

        score, intra_weight, total_weight = compute_hard_score(
            node_ids,
            assignments,
            data.edge_index,
            data.edge_weight
        )

    print("\nFinal hard score:")
    print("Intra-cluster weight:", intra_weight)
    print("Total edge weight:", total_weight)
    print("S(G,2):", score)

    print("\nFirst 10 hard assignments:")
    print(assignments[:10])

    print("\nFirst 10 hard one-hot vectors:")
    print(hard_one_hot[:10])

    num_cluster_0 = (assignments == 0).sum().item()
    num_cluster_1 = (assignments == 1).sum().item()

    print("\nCluster sizes:")
    print("Cluster 0:", num_cluster_0)
    print("Cluster 1:", num_cluster_1)

    results = {
        "score": score,
        "intra_cluster_weight": intra_weight,
        "total_edge_weight": total_weight,
        "loss_history": loss_history,
        "assignments": [],
        "hard_one_hot_assignments": []
    }

    for node_id, cluster, one_hot_vec in zip(
        node_ids,
        assignments.tolist(),
        hard_one_hot.tolist()
    ):
        results["assignments"].append({
            "node_id": node_id,
            "cluster": cluster
        })

        results["hard_one_hot_assignments"].append({
            "node_id": node_id,
            "one_hot": one_hot_vec
        })

    with open("level6/k2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to level6/k2_results.json")
    print("Saved loss graph to level6/loss_curve.png")