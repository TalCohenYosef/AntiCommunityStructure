import os
import sys
import torch
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from level3.load_gnn_data import load_gnn_input
from level4.model import AntiCommunityGNN


def compute_hard_score(node_ids, assignments, edge_index, edge_weight):
    """
    Compute the hard anti-community score S(G,2)
    using the hard assignments from argmax.
    """
    intra_weight = 0.0
    total_weight = 0.0

    src = edge_index[0]
    dst = edge_index[1]

    # because the graph was stored as both directions,
    # we count only one direction: u < v
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


def soft_anticommunity_loss(p, edge_index, edge_weight, lambda_balance=1000.0):
    """
    Soft anti-community loss + balance regularization.
    """
    src = edge_index[0]
    dst = edge_index[1]

    # main anti-community term
    dot_products = (p[src] * p[dst]).sum(dim=1)
    edge_loss = (edge_weight * dot_products).sum()

    # balance term: encourage average cluster usage to be close to [0.5, 0.5]
    mean_p = p.mean(dim=0)
    target = torch.tensor([0.5, 0.5], dtype=p.dtype, device=p.device)
    balance_loss = ((mean_p - target) ** 2).sum()

    total_loss = edge_loss + lambda_balance * balance_loss
    return total_loss


if __name__ == "__main__":
    data, node_ids = load_gnn_input("level2/gnn_input.json")

    model = AntiCommunityGNN(in_channels=2, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        p = model(data.x, data.edge_index, data.edge_weight)
        loss = soft_anticommunity_loss(
    p,
    data.edge_index,
    data.edge_weight,
    lambda_balance=1000.0
)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    print("\nTraining finished.")

    model.eval()
    with torch.no_grad():
        p = model(data.x, data.edge_index, data.edge_weight)
        assignments = torch.argmax(p, dim=1)

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

    print("First 10 hard assignments:")
    print(assignments[:10])

    results = {
        "score": score,
        "intra_cluster_weight": intra_weight,
        "total_edge_weight": total_weight,
        "assignments": []
    }

    for node_id, cluster in zip(node_ids, assignments.tolist()):
        results["assignments"].append({
            "node_id": node_id,
            "cluster": cluster
        })

    with open("level6/k2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to level6/k2_results.json")

    print("First 10 hard assignments:")
    print(assignments[:10])
    num_cluster_0 = (assignments == 0).sum().item()
    num_cluster_1 = (assignments == 1).sum().item()

    print("\nCluster sizes:")
    print("Cluster 0:", num_cluster_0)
    print("Cluster 1:", num_cluster_1)