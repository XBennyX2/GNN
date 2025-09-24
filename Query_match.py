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

