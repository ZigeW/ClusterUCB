from pykeops.torch import LazyTensor
import torch
import time
import random
import argparse
from collections import OrderedDict
import json

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"


def build_grad_lib(args):
    total_grad_dict = OrderedDict()
    for i, train_file in enumerate(args.train_file_names):
        train_file_grad = torch.load(args.grad_path.format(train_file))
        with open(args.train_file_path.format(train_file), 'r') as f:
            for j, line in enumerate(f.readlines()):
                data = json.loads(line)
                total_grad_dict[f"{data['dataset']}//{data['id']}//{j}"] = train_file_grad[j]

    return list(total_grad_dict.keys()), torch.vstack(list(total_grad_dict.values()))


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c_ids = random.sample(range(N), K)  # Random initialization for the centroids
    c = x[c_ids, :].clone()
    # Normalize the centroids for the cosine similarity:
    # c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_names', type=str, nargs='+', help='The name of the training file')
    parser.add_argument('--train_file_path', type=str, help='The path of the training data file')

    parser.add_argument("--grad_path", type=str, default=None, help="The path of grads used for selection")
    parser.add_argument("--output_path", type=str, default=None, help="The path to the output")

    parser.add_argument("--k", type=int, default=None, help="The number of clusters")
    parser.add_argument("--iters", type=int, default=20, help="The number of iterations in kmeans")

    parser.add_argument("--seed", type=int, default=42, help="The random seed")

    args = parser.parse_args()

    total_candidate, total_grads = build_grad_lib(args)
    random.seed(args.seed)
    cl, c = KMeans_cosine(total_grads, K=args.k, Niter=args.iters)
    torch.save([cl, c], args.output_path)
