import json
import torch
from torch_geometric.data import Data


def load_gnn_input(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_ids = data["node_ids"]
    x_init = data["x_init"]
    edge_index = data["edge_index"]
    edge_weight = data["edge_weight"]

    x = torch.tensor(x_init, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    return pyg_data, node_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load GNN Input Data")
    parser.add_argument("--input-dir", type=str, default="level2",
                       help="Input directory containing gnn_input.json")
    
    args = parser.parse_args()
    
    json_path = f"{args.input_dir}/gnn_input.json"
    
    data, node_ids = load_gnn_input(json_path)

    print("PyG Data loaded successfully")
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Feature matrix shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Edge weight shape:", data.edge_weight.shape)

    print("\nFirst 5 node ids:")
    for node_id in node_ids[:5]:
        print(node_id)

    print("\nFirst 5 feature vectors:")
    print(data.x[:5])

    print("\nFirst 10 directed edges:")
    print(data.edge_index[:, :10])

    print("\nFirst 10 edge weights:")
    print(data.edge_weight[:10])