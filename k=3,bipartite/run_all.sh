#!/usr/bin/env bash
set -e

# ============================================================================
# Anti-Community GNN Pipeline - Flexible for k-partite graphs
# ============================================================================
#
# Usage:
#   ./run_all.sh                                    # Bipartite (k=2)
#   ./run_all.sh --dim 3 --edges level1/edges_3partite.txt --output-dir level2_3partite
#   ./run_all.sh --dim 4 --edges level1/edges_4partite.txt --output-dir level2_4partite
#
# ============================================================================

# Default parameters
DIM=3
EDGES="level1/edges.txt"
NODES="level1/nodes.txt"
OUTPUT_DIR="level2"
SEED=42

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dim)
            DIM="$2"
            shift 2
            ;;
        --edges)
            EDGES="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Anti-Community GNN Pipeline"
            echo ""
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dim <int>          Vector dimension (2 for bipartite, 3 for 3-partite)"
            echo "  --edges <path>       Path to edges file"
            echo "  --nodes <path>       Path to nodes file"
            echo "  --output-dir <dir>   Output directory for level2 generated files"
            echo "  --seed <int>         Random seed"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_all.sh                                    # Bipartite (default)"
            echo "  ./run_all.sh --dim 3 --edges level1/edges_3partite.txt --output-dir level2_3partite"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
echo "=========================================================================="
echo "🚀 ANTI-COMMUNITY GNN PIPELINE"
echo "=========================================================================="
echo "📥 Configuration:"
echo "  • Dimension: ${DIM}"
echo "  • Nodes file: ${NODES}"
echo "  • Edges file: ${EDGES}"
echo "  • Output directory: ${OUTPUT_DIR}"
echo "  • Seed: ${SEED}"
echo "=========================================================================="
echo ""

# Step 1: Generate GNN input
echo "▶️  Step 1: Generating GNN input..."
python level2/generate_input_to_gnn.py \
    --nodes "${NODES}" \
    --edges "${EDGES}" \
    --dim "${DIM}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"
echo ""

# Step 2: Load GNN data
echo "▶️  Step 2: Loading GNN data..."
python level3/load_gnn_data.py --input-dir "${OUTPUT_DIR}"
echo ""

# Step 3: Test model
echo "▶️  Step 3: Testing model..."
python level5/test_model.py --input-dir "${OUTPUT_DIR}"
echo ""

# Step 4: Train k-partite model
echo "▶️  Step 4: Training k-partite model (k=${DIM})..."
python level6/train_k2.py --k "${DIM}" --input-dir "${OUTPUT_DIR}"
echo ""

# Step 5: Show results
echo "▶️  Step 5: Displaying results..."
python level7/show_results_gui.py
echo ""

echo "=========================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================================================="
echo ""