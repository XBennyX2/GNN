import argparse
import os
import pickle
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

from subgraph_matching.train import build_model
from subgraph_matching.alignment import gen_alignment_matrix

def main():
    parser = argparse.ArgumentParser(description="Query subgraph matching")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query graph pickle")
    parser.add_argument("--target_path", type=str, required=True, help="Path to target graph pickle")
    parser.add_argument("--model_path", type=str, default="ckpt/model.pt", help="Path to pretrained model")
    parser.add_argument("--method_type", type=str, default="order", choices=["order", "mlp"], help="Model type")
    parser.add_argument("--output_path", type=str, default="results/query_result.txt", help="Path to save True/False result")
    args = parser.parse_args()

    # Load graphs
    with open(args.query_path, "rb") as f:
        query = pickle.load(f)
    with open(args.target_path, "rb") as f:
        target = pickle.load(f)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Load pretrained model
    model = build_model(None)  # build_model supports loading default config
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Compute alignment matrix
    mat = gen_alignment_matrix(model, query, target, method_type=args.method_type)

    # Save alignment matrix plot
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.savefig("plots/alignment_matrix.png")
    print("Saved alignment matrix plot to plots/alignment_matrix.png")

    # Simple threshold to decide subgraph existence
    threshold = 0.5  # adjust if needed
    is_subgraph = np.any(mat > threshold)

    # Save result
    with open(args.output_path, "w") as f:
        f.write(str(is_subgraph))
    print(f"Query result saved to {args.output_path}: {is_subgraph}")

if __name__ == "__main__":
    main()
