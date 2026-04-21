#!/usr/bin/env bash
set -e

echo "Running generate_input_to_gnn.py"
python level2/generate_input_to_gnn.py

echo "Running load_gnn_data.py"
python level3/load_gnn_data.py

echo "Running test_model.py"
python level5/test_model.py

echo "Running train_k2.py"
python level6/train_k2.py

echo "Running show_results_gui.py"
python level7/show_results_gui.py