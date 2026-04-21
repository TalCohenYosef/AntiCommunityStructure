import random
import json


def load_nodes_with_random_one_hot(input_path, seed=42):
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

            node_id, node_type = line.split("\t")

            vec = random.choice([[1, 0], [0, 1]])

            node_ids.append(node_id)
            node_to_idx[node_id] = idx
            x_init.append(vec)

    return node_ids, node_to_idx, x_init


def save_nodes_with_init(output_path, node_ids, x_init):
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("node_id\tx1\tx2\n")
        for node_id, vec in zip(node_ids, x_init):
            f_out.write(f"{node_id}\t{vec[0]}\t{vec[1]}\n")


def load_edges(edges_path, node_to_idx):
    edge_index = []
    edge_weight = []

    with open(edges_path, "r", encoding="utf-8") as f_in:
        next(f_in)  # skip header

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            source, target, weight, roles = line.split("\t")

            if source not in node_to_idx or target not in node_to_idx:
                continue

            u = node_to_idx[source]
            v = node_to_idx[target]
            w = float(weight)

            # אם הגרף לא מכוון, נשמור את שני הכיוונים
            edge_index.append([u, v])
            edge_index.append([v, u])

            edge_weight.append(w)
            edge_weight.append(w)

    return edge_index, edge_weight


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
    nodes_input_path = "level1/nodes.txt"
    edges_input_path = "level1/edges.txt"

    nodes_output_path = "level2/nodes_with_init.txt"
    gnn_output_path = "level2/gnn_input.json"

    node_ids, node_to_idx, x_init = load_nodes_with_random_one_hot(
        nodes_input_path,
        seed=42
    )

    save_nodes_with_init(nodes_output_path, node_ids, x_init)

    edge_index, edge_weight = load_edges(edges_input_path, node_to_idx)

    save_gnn_input_json(
        gnn_output_path,
        node_ids,
        x_init,
        edge_index,
        edge_weight
    )

    print(f"Saved initialized nodes to: {nodes_output_path}")
    print(f"Saved GNN input to: {gnn_output_path}")
    print(f"Number of nodes: {len(node_ids)}")
    print(f"Number of directed edges in edge_index: {len(edge_index)}")