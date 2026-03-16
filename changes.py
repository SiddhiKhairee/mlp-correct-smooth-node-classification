import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.nn import CorrectAndSmooth
from torch_sparse import SparseTensor
import gc
import copy
import numpy as np

# CRITICAL: Fix unpickling error for PyTorch 2.6+
from functools import wraps
_original_torch_load = torch.load
@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
print("PyTorch load patch applied for OGB compatibility")

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# ==========================================
# 1. Model Definition
# ==========================================
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

# ==========================================
# 2. Helper: Weighted Sampler
# ==========================================
def get_weighted_loader(x, y, train_idx, batch_size=4096):
    print("   [Data] Creating Weighted Sampler...")
    y_train = y[train_idx].squeeze()
    
    # Inverse Frequency Weighting
    class_counts = torch.bincount(y_train)
    print(f"   [Data] Class distribution: min={class_counts.min()}, max={class_counts.max()}, mean={class_counts.float().mean():.1f}")
    
    class_weights = 1.0 / (class_counts.float() + 1e-5)
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Simple Dataset Wrapper to save memory (avoiding PyG Data object overhead in loader)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
        def __len__(self): return len(self.y)
        def __getitem__(self, idx): return self.x[idx], self.y[idx]

    dataset = SimpleDataset(x[train_idx], y[train_idx])
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

# ==========================================
# 3. Training & Inference Functions
# ==========================================
def train_model(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device).squeeze()
        optimizer.zero_grad()
        out = model(x_batch)
        loss = F.nll_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def get_predictions(model, x, device, chunk_size=20000):
    model.eval()
    preds = []
    for i in range(0, x.size(0), chunk_size):
        chunk = x[i:i+chunk_size].to(device)
        out = model(chunk)
        # We need probabilities for C&S (not log probabilities)
        preds.append(out.exp().cpu()) 
    return torch.cat(preds, dim=0)

@torch.no_grad()
def evaluate(y_pred, y_true, split_idx, evaluator, split_name='test'):
    """Helper function to evaluate accuracy"""
    acc = evaluator.eval({
        'y_true': y_true[split_idx[split_name]], 
        'y_pred': y_pred[split_idx[split_name]]
    })['acc']
    return acc

# ==========================================
# 4. Iterative C&S Function
# ==========================================
def iterative_c_s(base_probs, y_train, train_idx, adj_t, num_classes, steps=2, 
                  num_correction_layers=50, correction_alpha=1.0,
                  num_smoothing_layers=50, smoothing_alpha=0.8):
    """
    Iterative Correct & Smooth
    
    Each iteration:
    1. Correct: Propagate residual errors from training nodes
    2. Smooth: Propagate labels across graph
    
    Returns: Smoothed probabilities [num_nodes, num_classes]
    """
    print(f"   [C&S] Starting Iterative C&S ({steps} steps)...")
    print(f"   [C&S] Correction layers={num_correction_layers}, alpha={correction_alpha}")
    print(f"   [C&S] Smoothing layers={num_smoothing_layers}, alpha={smoothing_alpha}")
    
    # Setup C&S
    post = CorrectAndSmooth(
        num_correction_layers=num_correction_layers, 
        correction_alpha=correction_alpha,
        num_smoothing_layers=num_smoothing_layers, 
        smoothing_alpha=smoothing_alpha,
        autoscale=True, 
        scale=20.0
    )

    # Start with base predictions
    y_soft = base_probs.clone()
    
    # Iterative refinement
    for i in range(steps):
        print(f"     -> C&S Iteration {i+1}/{steps}")
        
        # Step A: Correct (Error Propagation on training nodes)
        y_soft = post.correct(y_soft, y_train[train_idx], train_idx, adj_t)
        
        # Step B: Smooth (Label Propagation)
        y_soft = post.smooth(y_soft, y_train[train_idx], train_idx, adj_t)
        
    print(f"   [C&S] Completed {steps} iterations")
    return y_soft

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Ensemble MLP with Weighted Sampling and Iterative C&S')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_models', type=int, default=3, help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per model')
    parser.add_argument('--c_s_steps', type=int, default=2, help='Number of C&S iterations')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Starting Ensemble Training on {device}")
    print(f"Configuration:")
    print(f"  - Ensemble size: {args.num_models} models")
    print(f"  - Epochs per model: {args.epochs}")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - C&S iterations: {args.c_s_steps}")
    print(f"  - Weighted Sampling: ENABLED")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("=== Loading Dataset ===")
    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    x = data.x
    y = data.y
    
    print(f"Dataset: ogbn-products")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Features: {x.size(1)}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train nodes: {len(split_idx['train']):,}")
    print(f"  Valid nodes: {len(split_idx['valid']):,}")
    print(f"  Test nodes: {len(split_idx['test']):,}")
    
    # 2. Prepare Graph for C&S (SparseTensor)
    print("\n=== Preparing Graph Structure ===")
    adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                         sparse_sizes=(data.num_nodes, data.num_nodes))
    adj_t = adj_t.to_symmetric()
    print(f"  Created symmetric adjacency matrix")
    
    # Free up memory!
    del data.edge_index
    del data
    gc.collect()

    # 3. Create Weighted Loader
    print("\n=== Creating Weighted Data Loader ===")
    loader = get_weighted_loader(x, y, split_idx['train'], batch_size=args.batch_size)
    
    # 4. Train Ensemble
    print(f"\n=== Training Ensemble ({args.num_models} Models) ===")
    evaluator = Evaluator(name='ogbn-products')
    
    all_preds = []
    
    for model_idx in range(args.num_models):
        print(f"\n[Model {model_idx+1}/{args.num_models}] Initializing...")
        model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes, 
                    num_layers=3, dropout=args.dropout).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Train Loop
        print(f"[Model {model_idx+1}/{args.num_models}] Training...")
        for epoch in range(1, args.epochs + 1):
            loss = train_model(model, loader, optimizer, device)
            if epoch % 20 == 0 or epoch == 1:
                print(f"   Epoch {epoch:3d}: Loss {loss:.4f}")
        
        # Get Predictions
        print(f"[Model {model_idx+1}/{args.num_models}] Generating predictions...")
        preds = get_predictions(model, x, device)
        
        # Quick evaluation of this model
        y_pred_single = preds.argmax(dim=-1, keepdim=True)
        acc_single = evaluate(y_pred_single, y, split_idx, evaluator, 'test')
        print(f"[Model {model_idx+1}/{args.num_models}] Individual Test Accuracy: {100*acc_single:.2f}%")
        
        all_preds.append(preds)
        
        # Delete model to free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Ensemble Aggregation (Before C&S)
    print(f"\n{'='*60}")
    print("=== Ensemble Aggregation ===")
    avg_probs = torch.stack(all_preds).mean(dim=0)
    
    # Evaluate Base Ensemble (without C&S)
    y_pred_ensemble = avg_probs.argmax(dim=-1, keepdim=True)
    
    train_acc = evaluate(y_pred_ensemble, y, split_idx, evaluator, 'train')
    valid_acc = evaluate(y_pred_ensemble, y, split_idx, evaluator, 'valid')
    test_acc = evaluate(y_pred_ensemble, y, split_idx, evaluator, 'test')
    
    print(f"Base Ensemble Results (WITHOUT C&S):")
    print(f"  Train Accuracy: {100*train_acc:.2f}%")
    print(f"  Valid Accuracy: {100*valid_acc:.2f}%")
    print(f"  Test Accuracy:  {100*test_acc:.2f}%")
    
    # Free memory
    del all_preds
    gc.collect()

    # 6. Iterative C&S
    print(f"\n{'='*60}")
    print("=== Applying Iterative Correct & Smooth ===")
    
    final_probs = iterative_c_s(
        avg_probs, y, split_idx['train'], adj_t, 
        dataset.num_classes, steps=args.c_s_steps
    )
    
    # 7. Final Evaluation (After C&S)
    final_pred = final_probs.argmax(dim=-1, keepdim=True)
    
    final_train_acc = evaluate(final_pred, y, split_idx, evaluator, 'train')
    final_valid_acc = evaluate(final_pred, y, split_idx, evaluator, 'valid')
    final_test_acc = evaluate(final_pred, y, split_idx, evaluator, 'test')
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Ensemble + Weighted Sampling + Iterative C&S)")
    print(f"{'='*60}")
    print(f"  Train Accuracy: {100*final_train_acc:.2f}%")
    print(f"  Valid Accuracy: {100*final_valid_acc:.2f}%")
    print(f"  Test Accuracy:  {100*final_test_acc:.2f}%")
    print(f"{'='*60}")
    
    print(f"\nImprovement from C&S:")
    print(f"  Train: {100*(final_train_acc - train_acc):.2f}% points")
    print(f"  Valid: {100*(final_valid_acc - valid_acc):.2f}% points")
    print(f"  Test:  {100*(final_test_acc - test_acc):.2f}% points")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()