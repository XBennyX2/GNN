# query_match.py
import argparse
import pickle
import os
import matplotlib.pyplot as plt
from subgraph_matching.alignment import gen_alignment_matrix
from subgraph_matching.train import build_model
from subgraph_matching.config import parse_encoder
from common import utils

THRESHOLD = 0.5  # decide subgraph existence

def query_subgraph(model, query, target, method_type="order", visualize=False):
    mat = gen_alignment_matrix(model, query, target, method_type=method_type)
    exists = mat.max() >= THRESHOLD
    if visualize:
        os.makedirs("plots", exist_ok=True)
        plt.imshow(mat, interpolation="nearest")
        plt.colorbar()
        plt.savefig("plots/query_alignment.png")
        print("Saved query alignment plot at plots/query_alignment.png")
    return exists

def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument("--query_path", required=True, help="Query graph pickle")
    parser.add_argument("--target_path", required=True, help="Target graph pickle")
    parser.add_argument("--method_type", default="order", help="order/mlp")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    args.test = True

    with open(args.query_path, "rb") as f:
        query = pickle.load(f)
    with open(args.target_path, "rb") as f:
        target = pickle.load(f)

    model = build_model(args)
    result = query_subgraph(model, query, target, args.method_type, args.visualize)
    print("Query exists as subgraph:", result)

if __name__ == "__main__":
    main()
