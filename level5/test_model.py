import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from level3.load_gnn_data import load_gnn_input
from level4.model import AntiCommunityGNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anti-Community GNN Model")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="level2",
        help="Input directory containing gnn_input.json"
    )

    args = parser.parse_args()

    json_path = f"{args.input_dir}/gnn_input.json"

    print(f"Loading data from: {json_path}")
    data, node_ids = load_gnn_input(json_path)

    actual_dim = data.x.shape[1]
    print(f"Detected input dimension from data.x: {actual_dim}")

    model = AntiCommunityGNN(
        in_channels=actual_dim,
        hidden_channels=8,
        out_channels=actual_dim
    )

    print("Running model...")
    p = model(data.x, data.edge_index, data.edge_weight)

    print("Output shape:", p.shape)
    print("First 5 output vectors:")
    print(p[:5])