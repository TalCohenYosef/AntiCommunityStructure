import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from level3.load_gnn_data import load_gnn_input
from level4.model import AntiCommunityGNN

data, node_ids = load_gnn_input("level2/gnn_input.json")

model = AntiCommunityGNN(in_channels=2, hidden_channels=8, out_channels=2)
p = model(data.x, data.edge_index, data.edge_weight)

print("Output shape:", p.shape)
print("First 5 output vectors:")
print(p[:5])