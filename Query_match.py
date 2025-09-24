import argparse
import pickle
import networkx as nx
from subgraph_matching.train import build_model
from subgraph_matching.alignment import gen_alignment_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='Subgraph Query Match')
    parser.add_argument('--query_path', type=str, default="examples/query_graph.pkl",
                        help='Path to query graph pickle')
    parser.add_argument('--target_path', type=str, default="examples/target_graph.pkl",
                        help='Path to target graph pickle')
    parser.add_argument('--model_path', type=str, default="ckpt/model.pt",
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--method_type', type=str, default="order",
                        help='Method type: "order" or "mlp"')
    parser.add_argument('--output_path', type=str, default="results/query_result.txt",
                        help='File to save query result')
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Load graphs
    with open(args.query_path, "rb") as f:
        query = pickle.load(f)
    with open(args.target_path, "rb") as f:
        target = pickle.load(f)

    # Build model with proper args
    args = parser.parse_args()
    model = build_model(args)


    # Generate alignment matrix
    mat = gen_alignment_matrix(model, query, target, method_type=args.method_type)

    # Decide subgraph match (example: max confidence > threshold)
    threshold = 0.5
    is_subgraph = np.max(mat) > threshold

    # Save result
    with open(args.output_path, "w") as f:
        f.write(str(is_subgraph))

    # Save plot
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.title("Query vs Target Alignment Matrix")
    plt.savefig("plots/query_alignment.png")
    plt.close()

    print(f"Subgraph match result: {is_subgraph}")
    print(f"Alignment matrix plot saved in plots/query_alignment.png")

if __name__ == "__main__":
    main()
