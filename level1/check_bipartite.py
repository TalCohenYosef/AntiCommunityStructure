import networkx as nx
import pandas as pd
from datetime import datetime

def check_and_analyze_bipartite(nodes_file='nodes.txt', edges_file='edges.txt', output_file='bipartite_analysis.txt'):
    """
    Check if the graph is bipartite and print the sizes of each class.
    Save results to a text file.
    
    Args:
        nodes_file: Path to nodes.txt file
        edges_file: Path to edges.txt file
        output_file: Path to output text file
    """
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Load nodes
        nodes_df = pd.read_csv(nodes_file, sep='\t')
        f.write("=" * 60 + "\n")
        f.write("BIPARTITE GRAPH ANALYSIS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    
        # Create graph
        G = nx.Graph()
        
        # Add nodes with their type
        node_types = {}
        for _, row in nodes_df.iterrows():
            node_id = row['node_id']
            node_type = row['type']
            G.add_node(node_id, type=node_type)
            node_types[node_id] = node_type
        
        # Load edges
        edges_df = pd.read_csv(edges_file, sep='\t')
        
        # Add edges
        for _, row in edges_df.iterrows():
            source = row['source']
            target = row['target']
            weight = row['weight']
            roles = row['roles']
            G.add_edge(source, target, weight=weight, roles=roles)
        
        # Check if bipartite
        is_bipartite = nx.is_bipartite(G)
        
        f.write(f"\n✓ Is the graph bipartite? {is_bipartite}\n")
        
        if is_bipartite:
            # Get the two partitions
            partition = nx.bipartite.sets(G)
            class1, class2 = list(partition)
            
            f.write(f"\n📊 CLASS SIZES:\n")
            f.write(f"  Class 1 size: {len(class1)}\n")
            f.write(f"  Class 2 size: {len(class2)}\n")
            f.write(f"  Total nodes: {len(G.nodes())}\n")
            
            # Analyze by type
            f.write(f"\n📋 NODE TYPES DISTRIBUTION:\n")
            types_count = nodes_df['type'].value_counts()
            for node_type, count in types_count.items():
                f.write(f"  {node_type}: {count}\n")
            
            # Show which types are in which partition
            f.write(f"\n🔍 TYPE DISTRIBUTION PER CLASS:\n")
            
            class1_types = {}
            for node in class1:
                node_type = node_types.get(node, 'unknown')
                class1_types[node_type] = class1_types.get(node_type, 0) + 1
            
            class2_types = {}
            for node in class2:
                node_type = node_types.get(node, 'unknown')
                class2_types[node_type] = class2_types.get(node_type, 0) + 1
            
            f.write(f"\n  Class 1:\n")
            for node_type, count in sorted(class1_types.items()):
                f.write(f"    {node_type}: {count}\n")
            
            f.write(f"\n  Class 2:\n")
            for node_type, count in sorted(class2_types.items()):
                f.write(f"    {node_type}: {count}\n")
            
            # Graph statistics
            f.write(f"\n📈 GRAPH STATISTICS:\n")
            f.write(f"  Total edges: {len(G.edges())}\n")
            f.write(f"  Total nodes: {len(G.nodes())}\n")
            f.write(f"  Density: {nx.density(G):.4f}\n")
            
        else:
            f.write("The graph is NOT bipartite!\n")
            f.write(f"Total nodes: {len(G.nodes())}\n")
            f.write(f"Total edges: {len(G.edges())}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        print(f"✓ Analysis complete! Results saved to: {output_file}")
    
        return G, is_bipartite

if __name__ == "__main__":
    G, is_bipartite = check_and_analyze_bipartite()
