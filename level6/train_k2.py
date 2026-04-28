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


def compute_hard_score(assignments, edge_index, edge_weight):
    """
    Old clean anti-community score:
    S(G,k) = 1 - intra_weight / total_weight
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

def hard_empty_penalty_st(p, min_nodes_per_cluster=20):
    """
    Strong empty-cluster penalty using hard argmax sizes,
    with straight-through estimator.
    """

    n, k = p.shape

    hard_assignments = torch.argmax(p, dim=1)
    hard_one_hot = F.one_hot(hard_assignments, num_classes=k).float()

    # forward behaves like hard_one_hot, backward behaves like p
    z = hard_one_hot + p - p.detach()

    cluster_sizes = z.sum(dim=0)

    shortage = torch.relu(min_nodes_per_cluster - cluster_sizes)

    penalty = (shortage ** 2).sum()

    return penalty

def differentiable_sep_penalty(p, edge_index, edge_weight, epsilon=0.001):
    """
    Differentiable - gradients flow through p directly.
    """
    src = edge_index[0]
    dst = edge_index[1]
    mask = src < dst
    src, dst = src[mask], dst[mask]
    ew = edge_weight[mask]
    total_weight = ew.sum()

    k = p.shape[1]
    min_between = None

    for a in range(k):
        for b in range(a + 1, k):
            between_ab = (ew * p[src, a] * p[dst, b] + ew * p[src, b] * p[dst, a]).sum()
            between_ab_normalized = between_ab / total_weight
            if min_between is None:
                min_between = between_ab_normalized
            else:
                # soft minimum - differentiable
                min_between = torch.minimum(min_between, between_ab_normalized)

    return 1.0 / (epsilon + min_between)




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

    epochs = 700
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        p = model(data.x, data.edge_index, data.edge_weight)

        loss = soft_anticommunity_loss(p, original_edge_index, original_edge_weight)
        sep = differentiable_sep_penalty(p, original_edge_index, original_edge_weight)
        empty = hard_empty_penalty_st(p,min_nodes_per_cluster=3)

        total_loss = loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(total_loss.item())
        if epoch % 10 == 0:
            assignments = torch.argmax(p, dim=1)
            cluster_sizes = [(assignments == i).sum().item() for i in range(args.k)]
            print(f"Epoch {epoch:03d} | loss={loss.item():.1f} | clusters={cluster_sizes}")

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