def parse_wiki_elections_to_3partite(
    input_file,
    output_nodes_file="nodes_3partite.txt",
    output_edges_file="edges_3partite.txt"
):
    nodes = set()
    edges = []
    election_counter = 0

    current_candidate = None
    current_nominator = None
    current_voters = []

    def add_node(node_id, node_type):
        nodes.add((node_id, node_type))

    def flush_current_election():
        nonlocal current_candidate, current_nominator, current_voters

        if current_candidate is None:
            return

        if current_nominator is not None:
            edges.append((current_nominator, current_candidate, 1))

        for voter_node, vote_value in current_voters:
            edges.append((voter_node, current_candidate, vote_value))

        if current_nominator is not None:
            for voter_node, _ in current_voters:
                edges.append((current_nominator, voter_node, 1))

        current_candidate = None
        current_nominator = None
        current_voters = []

    with open(input_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            tag = parts[0]

            if tag == "E":
                if current_candidate is not None:
                    flush_current_election()
                election_counter += 1

            elif tag == "U":
                if len(parts) >= 3:
                    user_id = parts[1]
                    node_id = f"cand_{user_id}"
                    add_node(node_id, "candidate")
                    current_candidate = node_id

            elif tag == "N":
                if len(parts) >= 3:
                    user_id = parts[1]
                    node_id = f"nom_{user_id}"
                    add_node(node_id, "nominator")
                    current_nominator = node_id

            elif tag == "V":
                if len(parts) >= 5:
                    vote_value = int(parts[1])
                    voter_user_id = parts[2]
                    voter_node = f"vote_{voter_user_id}"
                    add_node(voter_node, "voter")
                    current_voters.append((voter_node, vote_value))

    flush_current_election()

    with open(output_nodes_file, "w", encoding="utf-8") as f:
        f.write("node_id\ttype\n")
        for node_id, node_type in sorted(nodes):
            f.write(f"{node_id}\t{node_type}\n")

    with open(output_edges_file, "w", encoding="utf-8") as f:
        f.write("source\ttarget\tweight\n")
        for source, target, weight in edges:
            f.write(f"{source}\t{target}\t{1}\n")

    print("Done.")
    print(f"Nodes written to: {output_nodes_file}")
    print(f"Edges written to: {output_edges_file}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")


if __name__ == "__main__":
    input_path = "level1/raw_data.txt"
    parse_wiki_elections_to_3partite(input_path)