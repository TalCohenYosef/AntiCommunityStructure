from collections import defaultdict

EDGES_FILE = "edges.txt"
NODES_FILE = "nodes.txt"

OUT_EDGES_FILE = "small_edges.txt"
OUT_NODES_FILE = "small_nodes.txt"

TARGET_NODES = 1000

# -----------------------------
# Read nodes
# -----------------------------
node_type = {}

with open(NODES_FILE, "r", encoding="utf-8") as f:
    header = next(f)

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        node_id, node_t = parts[0], parts[1]
        node_type[node_id] = node_t

# -----------------------------
# Read edges and compute degrees
# -----------------------------
edges = []
degree = defaultdict(int)

with open(EDGES_FILE, "r", encoding="utf-8") as f:
    header = next(f)

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue

        source, target, weight = parts[:3]
        roles = parts[3] if len(parts) > 3 else ""

        edges.append((source, target, weight, roles))

        degree[source] += 1
        degree[target] += 1

# -----------------------------
# Select strongest nodes
# -----------------------------
selected_nodes = {
    node
    for node, deg in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:TARGET_NODES]
}

# -----------------------------
# Filter edges
# -----------------------------
filtered_edges = [
    (source, target, weight, roles)
    for source, target, weight, roles in edges
    if source in selected_nodes and target in selected_nodes
]

# Important:
# keep only nodes that actually appear in the filtered edges
final_nodes = set()

for source, target, weight, roles in filtered_edges:
    final_nodes.add(source)
    final_nodes.add(target)

# -----------------------------
# Save small nodes file
# -----------------------------
with open(OUT_NODES_FILE, "w", encoding="utf-8") as f:
    f.write("node_id\ttype\n")

    for node in sorted(final_nodes):
        f.write(f"{node}\t{node_type.get(node, 'unknown')}\n")

# -----------------------------
# Save small edges file
# -----------------------------
with open(OUT_EDGES_FILE, "w", encoding="utf-8") as f:
    f.write("source\ttarget\tweight\troles\n")

    for source, target, weight, roles in filtered_edges:
        f.write(f"{source}\t{target}\t{weight}\t{roles}\n")

print("Original nodes:", len(node_type))
print("Original edges:", len(edges))
print("Selected nodes:", len(selected_nodes))
print("Final nodes:", len(final_nodes))
print("Final edges:", len(filtered_edges))
print("Saved nodes to:", OUT_NODES_FILE)
print("Saved edges to:", OUT_EDGES_FILE)