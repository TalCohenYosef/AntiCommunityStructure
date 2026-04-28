import random
import json
import argparse
import sys


def load_nodes_with_random_one_hot(input_path, dim=2, seed=42):
    """
    Load nodes and assign random one-hot vectors.
    
    Args:
        input_path: Path to nodes.txt file
        dim: Dimensionality of the one-hot vectors (default: 2 for bipartite, 3 for 3-partite)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    node_ids = []
    node_to_idx = {}
    x_init = []

    with open(input_path, "r", encoding="utf-8") as f_in:
        next(f_in)  # skip header

        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            node_id = parts[0]

            # Create random one-hot vector
            # For dim=2: [1,0] or [0,1]
            # For dim=3: [1,0,0] or [0,1,0] or [0,0,1]
            vec = [0] * dim
            random_idx = random.randint(0, dim - 1)
            vec[random_idx] = 1
         
            node_to_idx[node_id] = len(node_ids)
            node_ids.append(node_id)
            x_init.append(vec)

    return node_ids, node_to_idx, x_init


def save_nodes_with_init(output_path, node_ids, x_init):
    dim = len(x_init[0]) if x_init else 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        header = ["node_id"] + [f"x{i+1}" for i in range(dim)]
        f_out.write("\t".join(header) + "\n")

        for node_id, vec in zip(node_ids, x_init):
            row = [node_id] + [str(v) for v in vec]
            f_out.write("\t".join(row) + "\n")


def load_edges(edges_path, node_to_idx):
    edge_index = []
    edge_weight = []

    with open(edges_path, "r", encoding="utf-8") as f_in:
        next(f_in)  # skip header

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            
            # Handle both 3-column (source, target, weight) and 4-column (source, target, weight, roles) formats
            if len(parts) == 3:
                source, target, weight = parts
                roles = "unknown"  # Default role for 3-column format
            elif len(parts) == 4:
                source, target, weight, roles = parts
            else:
                print(f"⚠️  Warning: Skipping malformed line: {line}")
                continue

            if source not in node_to_idx or target not in node_to_idx:
                continue

            u = node_to_idx[source]
            v = node_to_idx[target]
            w = float(weight)

            edge_index.append([u, v])
            edge_index.append([v, u])

            edge_weight.append(w)
            edge_weight.append(w)

    return edge_index, edge_weight

def add_random_edges(edge_index, edge_weight, num_nodes, percent, seed=42):
    random.seed(seed)

    num_existing_undirected = len(edge_index) // 2
    num_new_undirected = int(num_existing_undirected * percent)

    existing = set()
    for u, v in edge_index:
        existing.add((min(u, v), max(u, v)))

    new_edges_added = 0

    while new_edges_added < num_new_undirected:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u == v:
            continue

        key = (min(u, v), max(u, v))

        if key in existing:
            continue

        existing.add(key)

        edge_index.append([u, v])
        edge_index.append([v, u])

        edge_weight.append(1.0)
        edge_weight.append(1.0)

        new_edges_added += 1

    return edge_index, edge_weight, new_edges_added

def save_gnn_input_json(output_path, node_ids, x_init, edge_index, edge_weight):
    data = {
        "node_ids": node_ids,
        "x_init": x_init,
        "edge_index": edge_index,
        "edge_weight": edge_weight
    }

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GNN input from graph data (flexible for bipartite, 3-partite, etc.)"
    )
    
    parser.add_argument(
        "--nodes",
        type=str,
        default="level1/nodes.txt",
        help="Path to nodes.txt file (default: level1/nodes.txt)"
    )
    
    parser.add_argument(
        "--edges",
        type=str,
        default="level1/edges.txt",
        help="Path to edges.txt file (default: level1/edges.txt)"
    )
    
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimensionality of one-hot vectors (2 for bipartite, 3 for 3-partite, etc. - default: 2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="level2",
        help="Output directory for generated files (default: level2)"
    )
    
    parser.add_argument(
    "--random-edge-percent",
    type=float,
    default=0.0,
    help="Percent of random edges to add relative to existing undirected edges"
)
    args = parser.parse_args()
    
    # Validate inputs
    try:
        with open(args.nodes, "r") as f:
            pass
    except FileNotFoundError:
        print(f"❌ Error: Nodes file not found: {args.nodes}")
        sys.exit(1)
    
    try:
        with open(args.edges, "r") as f:
            pass
    except FileNotFoundError:
        print(f"❌ Error: Edges file not found: {args.edges}")
        sys.exit(1)
    
    if args.dim < 2:
        print(f"❌ Error: Dimension must be at least 2, got {args.dim}")
        sys.exit(1)
    
    # Set output paths
    nodes_output_path = f"{args.output_dir}/nodes_with_init.txt"
    gnn_output_path = f"{args.output_dir}/gnn_input.json"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔧 GENERATING GNN INPUT")
    print(f"{'='*70}")
    print(f"📥 Configuration:")
    print(f"  • Nodes file: {args.nodes}")
    print(f"  • Edges file: {args.edges}")
    print(f"  • Vector dimension: {args.dim}")
    print(f"  • Seed: {args.seed}")
    print(f"  • Output directory: {args.output_dir}")
    
    # Load and process data
    print(f"\n⚙️  Processing...")
    
    node_ids, node_to_idx, x_init = load_nodes_with_random_one_hot(
        args.nodes,
        dim=args.dim,
        seed=args.seed
    )
    print(f"  ✓ Loaded {len(node_ids)} nodes")

    save_nodes_with_init(nodes_output_path, node_ids, x_init)
    print(f"  ✓ Saved nodes with init vectors to: {nodes_output_path}")

    edge_index, edge_weight = load_edges(args.edges, node_to_idx)
    print(f"  ✓ Loaded {len(edge_index)} directed edges")


    if args.random_edge_percent > 0:
        edge_index, edge_weight, added = add_random_edges(
            edge_index,
            edge_weight,
            num_nodes=len(node_ids),
            percent=args.random_edge_percent,
            seed=args.seed
        )

        print(f"  ✓ Added {added} random undirected edges")
        print(f"  ✓ Total directed edges after addition: {len(edge_index)}")

    save_gnn_input_json(
        gnn_output_path,
        node_ids,
        x_init,
        edge_index,
        edge_weight
    )
    print(f"  ✓ Saved GNN input to: {gnn_output_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS!")
    print(f"{'='*70}\n")

    print(f"Saved initialized nodes to: {nodes_output_path}")
    print(f"Saved GNN input to: {gnn_output_path}")
    print(f"Number of nodes: {len(node_ids)}")
    print(f"Number of directed edges in edge_index: {len(edge_index)}")