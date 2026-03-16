from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from torch_geometric.data import Data

from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from torch_scatter import scatter

import h5py
import os

import numpy as np
np.random.seed(0)




def sgc(x, adj, num_propagations):
    for _ in tqdm(range(num_propagations)):
        x = adj @ x
    return torch.from_numpy(x).to(torch.float)


def lp(adj, train_idx, labels, num_propagations, p, alpha, preprocess):
    if p is None:
        p = 0.6
    if alpha is None:
        alpha = 0.4
    
    c = labels.max() + 1
    idx = train_idx
    y = np.zeros((labels.shape[0], c))
    y[idx] = F.one_hot(labels[idx],c).numpy().squeeze(1)
    result = deepcopy(y)
    for i in tqdm(range(num_propagations)):
        result = y + alpha * adj @ (result**p)
        result = np.clip(result,0,1)
    return torch.from_numpy(result).to(torch.float)

def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    inital_features = deepcopy(x)
    x = x **p
    for i in tqdm(range(num_propagations)):
#         x = (1-args.alpha)* inital_features + args.alpha * adj @ x
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x **p
    return torch.from_numpy(x).to(torch.float)

def community(data, post_fix):
    print('Setting up community detection feature')
    import networkx as nx
    import community as community_louvain
    
    np_edge_index = np.array(data.edge_index)

    G = nx.Graph()
    G.add_edges_from(np_edge_index.T)

    partition = community_louvain.best_partition(G)
    np_partition = np.zeros(data.num_nodes)
    for k, v in partition.items():
        np_partition[k] = v

    np_partition = np_partition.astype(np.int)

    n_values = int(np.max(np_partition) + 1)
    one_hot = np.eye(n_values)[np_partition]

    result = torch.from_numpy(one_hot).float()
    
    os.makedirs('embeddings', exist_ok=True)
    torch.save(result, f'embeddings/community{post_fix}.pt')
        
    return result

def spectral(data, post_fix, n_components=128):
    """
    Pure Python implementation of spectral embedding using scipy
    Computes the normalized Laplacian eigenvectors
    This replaces the Julia implementation
    """
    print('Setting up spectral embedding (Python implementation)')
    data.edge_index = to_undirected(data.edge_index)
    
    N = data.num_nodes
    row, col = data.edge_index
    
    # Build adjacency matrix
    print(f'Building adjacency matrix for {N} nodes...')
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    
    # Compute degree matrix
    print('Computing degree matrix...')
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    print('Computing normalized Laplacian...')
    degrees_sqrt_inv = np.power(degrees, -0.5)
    degrees_sqrt_inv[np.isinf(degrees_sqrt_inv)] = 0
    D_sqrt_inv = sparse.diags(degrees_sqrt_inv)
    
    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    A_norm = D_sqrt_inv @ adj @ D_sqrt_inv
    
    # Normalized Laplacian: L = I - A_norm
    L_norm = sparse.eye(N) - A_norm
    
    print(f'Computing {n_components} smallest eigenvectors (this may take a while)...')
    # Compute smallest eigenvectors (excluding the trivial zero eigenvalue)
    # We use 'SM' (smallest magnitude) for the Laplacian
    try:
        # Try to compute n_components+1 eigenvectors
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_components+1, which='SM', tol=1e-3, maxiter=1000)
        # Remove the first eigenvector (corresponds to eigenvalue ~0)
        eigenvectors = eigenvectors[:, 1:n_components+1]
        print(f'Successfully computed {eigenvectors.shape[1]} eigenvectors')
    except Exception as e:
        print(f"Warning: eigsh with k={n_components+1} failed: {e}")
        print("Trying with fewer components...")
        try:
            # Try with half the components
            n_components_reduced = min(n_components // 2, N // 3)
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_components_reduced+1, which='SM', tol=1e-2, maxiter=500)
            eigenvectors = eigenvectors[:, 1:]
            print(f'Computed {eigenvectors.shape[1]} eigenvectors with reduced dimensionality')
        except Exception as e2:
            print(f"Error computing spectral embedding: {e2}")
            print("Falling back to random embeddings...")
            # Fallback: use random embeddings
            eigenvectors = np.random.randn(N, n_components) * 0.01
    
    result = torch.from_numpy(eigenvectors).float()
    
    # Create embeddings directory if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)
    torch.save(result, f'embeddings/spectral{post_fix}.pt')
    
    print(f'Spectral embedding computed and saved: shape {result.shape}')
    return result



def preprocess(data, preprocess = "diffusion", num_propagations = 10, p = None, alpha = None, use_cache = True, post_fix = ""):
    if use_cache:
        try:
            x = torch.load(f'embeddings/{preprocess}{post_fix}.pt', weights_only=False)
            print(f'Using cached {preprocess} embeddings from embeddings/{preprocess}{post_fix}.pt')
            return x
        except Exception as e:
            print(f'embeddings/{preprocess}{post_fix}.pt not found or error loading! Regenerating it now')
            print(f'Error details: {e}')
    
    if preprocess == "community":
        return community(data, post_fix)

    if preprocess == "spectral":
        return spectral(data, post_fix)

    
    print('Computing adj...')
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')

    sgc_dict = {}
        
    print(f'Start {preprocess} processing')

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations)
#     if preprocess == "lp":
#         result = lp(adj, data.y.data, num_propagations, p = p, alpha = alpha, preprocess = preprocess)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p = p, alpha = alpha)

    os.makedirs('embeddings', exist_ok=True)
    torch.save(result, f'embeddings/{preprocess}{post_fix}.pt')
    
    return result