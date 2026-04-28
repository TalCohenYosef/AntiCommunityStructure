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
    Compute the hard anti-community score S(G,k)
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

    score = 1.0 - (intra_weight / total_weight)
    return score, intra_weight, total_weight


def soft_anticommunity_loss(p, edge_index, edge_weight):
    src = edge_index[0]
    dst = edge_index[1]
    mask = src < dst
    src = src[mask]
    dst = dst[mask]
    ew = edge_weight[mask]

    dot_products = (p[src] * p[dst]).sum(dim=1)
    loss = (ew * dot_products).sum() 
    return loss

def min_cluster_usage_penalty(p, min_fraction=0.01):
    """
    Penalizes clusters that are nearly or completely empty.
    min_fraction = minimum fraction of nodes a cluster should receive (default: 1%)
    
    Example: For 9000 nodes split 2000/7000:
        - Cluster with 22% -> no penalty
        - Cluster with 78% -> no penalty
        - Empty cluster 0% -> large penalty
    """
    mean_p = p.mean(dim=0)  # [k] - average probability mass per cluster
    
    # Only penalize clusters below min_fraction
    penalty = torch.relu(min_fraction - mean_p).sum()
    
    return penalty


def compute_pairwise_min_inter_cluster_weight(assignments, edge_index, edge_weight, k, epsilon=0.001):
    src = edge_index[0]
    dst = edge_index[1]

    pairwise_between = {}
    for a in range(k):
        for b in range(a + 1, k):
            pairwise_between[(a, b)] = 0.0

    total_weight = 0.0
    for i in range(edge_index.shape[1]):
        u = src[i].item()
        v = dst[i].item()
        if u > v:
            continue
        w = edge_weight[i].item()
        total_weight += w
        cluster_u = assignments[u].item()
        cluster_v = assignments[v].item()
        if cluster_u != cluster_v:
            a, b = sorted((cluster_u, cluster_v))
            pairwise_between[(a, b)] += w

    # normalize each pair by total weight
    min_between_normalized = min(v / total_weight for v in pairwise_between.values())

    penalty = 1.0 / (epsilon + min_between_normalized)  # range: [1/epsilon down to ~1]
    return penalty

def empty_cluster_penalty(assignments, k, num_nodes, epsilon=0.001):
    """
    Penalizes clusters with zero (or very few) nodes.
    Normalized by num_nodes, so range is [0, 1] per cluster.
    """
    penalty = 0.0
    for i in range(k):
        fraction = (assignments == i).sum().item() / num_nodes  # [0, 1]
        if fraction < epsilon:
            penalty += 1.0 / (epsilon + fraction)  # large when empty
    return penalty



def compute_two_hop_neighbors(edge_index, edge_weight, num_nodes):
    """
    Convert 1-hop edges to 2-hop neighbors.
    For each node, creates edges only to neighbors of neighbors (not to direct neighbors).
    
    Args:
        edge_index: [2, num_edges] tensor
        edge_weight: [num_edges] tensor
        num_nodes: total number of nodes
        
    Returns:
        new_edge_index: edges between 2-hop neighbors
        new_edge_weight: weights for 2-hop edges
    """
    import collections
    
    # Build adjacency list with weights
    adj = collections.defaultdict(dict)
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        w = edge_weight[i].item()
        
        # Store both directions (undirected graph)
        if v not in adj[u]:
            adj[u][v] = w
        if u not in adj[v]:
            adj[v][u] = w
    
    # Compute 2-hop edges
    two_hop_edges = {}
    for u in range(num_nodes):
        # Get direct neighbors of u
        direct_neighbors = set(adj[u].keys())
        
        # For each direct neighbor, get their neighbors
        for v in direct_neighbors:
            neighbors_of_v = adj[v].keys()
            for w in neighbors_of_v:
                # Skip if w is u or a direct neighbor of u
                if w != u and w not in direct_neighbors:
                    # Create edge (u, w) with combined weight
                    edge_key = (min(u, w), max(u, w))
                    weight_uv = adj[u][v]
                    weight_vw = adj[v][w]
                    
                    # Combine weights: multiply the two hop weights
                    combined_weight = weight_uv * weight_vw
                    
                    if edge_key not in two_hop_edges:
                        two_hop_edges[edge_key] = combined_weight
                    else:
                        # If multiple paths, take average
                        two_hop_edges[edge_key] = (two_hop_edges[edge_key] + combined_weight) / 2
    
    # Convert to edge_index and edge_weight tensors
    if len(two_hop_edges) == 0:
        print("Warning: No 2-hop edges found!")
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
        new_edge_weight = torch.zeros((0,), dtype=torch.float)
    else:
        edges = list(two_hop_edges.keys())
        weights = list(two_hop_edges.values())
        
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        
        # Add both directions for undirected graph
        new_edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        new_edge_weight = torch.tensor(weights + weights, dtype=torch.float)
    
    return new_edge_index, new_edge_weight


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Anti-Community GNN Model")
    parser.add_argument("--input-dir", type=str, default="level2",
                       help="Input directory containing gnn_input.json")
    parser.add_argument("--k", type=int, default=2,
                       help="Number of partitions (2 for bipartite, 3 for 3-partite)")
    
    args = parser.parse_args()
    
    json_path = f"{args.input_dir}/gnn_input.json"
    
    print(f"Loading data from: {json_path}")
    data, node_ids = load_gnn_input(json_path)
    
    # Save original edge index and weights for loss calculation
    original_edge_index = data.edge_index.clone()
    original_edge_weight = data.edge_weight.clone()
    print(f"Original graph has {original_edge_index.shape[1]} edges (direct neighbors)")
    
    # Convert to 2-hop neighbors for GCN
    print("Converting to 2-hop neighbors for GCN...")
    data.edge_index, data.edge_weight = compute_two_hop_neighbors(
        data.edge_index, 
        data.edge_weight, 
        data.x.shape[0]
    )
    print(f"GCN graph has {data.edge_index.shape[1]} edges (2-hop neighbors)")
 
    print(f"Creating model with k={args.k}")
    model = AntiCommunityGNN(in_channels=args.k, hidden_channels=8, out_channels=args.k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 800
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        p = model(data.x, data.edge_index, data.edge_weight)
        assignments = torch.argmax(p, dim=1)

        loss = soft_anticommunity_loss(p, original_edge_index, original_edge_weight)  # [0, 1]
        sep = compute_pairwise_min_inter_cluster_weight(assignments, original_edge_index, original_edge_weight, args.k)
        empty = empty_cluster_penalty(assignments, args.k, data.x.shape[0])

        total_loss = loss + sep + empty
        total_loss.backward()
        optimizer.step()

        loss_value = total_loss.item()
        loss_history.append(loss_value)
        if epoch % 10 == 0:
          cluster_sizes = [(assignments == i).sum().item() for i in range(args.k)]
          print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | sep={sep:.4f} | empty={empty:.4f} | total={total_loss.item():.4f} | clusters={cluster_sizes}")
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
        hard_one_hot = F.one_hot(assignments, num_classes=args.k).float()

        score, intra_weight, total_weight = compute_hard_score(
            node_ids,
            assignments,
            original_edge_index,
            original_edge_weight
        )

    print("\nFinal hard score:")
    print("Intra-cluster weight:", intra_weight)
    print("Total edge weight:", total_weight)
    print(f"S(G,{args.k}):", score)

    print("\nFirst 10 hard assignments:")
    print(assignments[:10])

    print("\nFirst 10 hard one-hot vectors:")
    print(hard_one_hot[:10])

    print("\nCluster sizes:")
    for i in range(args.k):
        num_cluster_i = (assignments == i).sum().item()
        print(f"Cluster {i}: {num_cluster_i}")

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