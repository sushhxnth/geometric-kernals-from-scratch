import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.operators import ExplicitGINLayer, ExplicitGCNLayer, dirichlet_energy
from src.kernels import compute_pairwise_wasserstein, ExplicitKernelRidge

def generate_synthetic_sbm_dataset(
    n_graphs: int,
    min_nodes: int = 12,
    max_nodes: int = 24,
    homophily: float = 0.75,
    noise_std: float = 0.15,
    seed: int = 42,
):
    
    rng = np.random.default_rng(seed)
    graphs = []
    targets = []
    sizes = []

    for _ in range(n_graphs):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        sizes.append(n)
        clusters = rng.integers(0, 2, size=n)

        # Edge probabilities governed by homophily
        p_in = 0.45 * homophily + 0.1
        p_out = 0.45 * (1.0 - homophily) + 0.05
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                p = p_in if clusters[i] == clusters[j] else p_out
                if rng.random() < p:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Correlated node features + controlled observation noise
        base_signal = (2 * clusters - 1)[:, None] * np.array([[1.0, -0.8, 0.5, 0.3]])
        x = (base_signal + rng.normal(0, 0.4, size=(n, 4))).astype(np.float32)

        # Target function depends on 2-hop homophilous motif interaction + nonlinear smoothness
        deg = adj.sum(axis=1, keepdims=True) + 1.0
        diffused = (adj / deg) @ x
        motif_energy = np.mean(np.sin(diffused[:, 0]) + 0.5 * (diffused[:, 1] ** 2))
        spectral_term = np.sum(adj) / (n ** 1.5)
        y = motif_energy + 0.8 * spectral_term + rng.normal(0, noise_std)

        graphs.append((adj, x))
        targets.append(float(y))

    return graphs, np.array(targets, dtype=np.float32), np.array(sizes)


class GraphLevelGINRegressor(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 24):
        super().__init__()
        self.gin1 = ExplicitGINLayer(in_dim, hidden_dim)
        self.gin2 = ExplicitGINLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gin1(x, adj))
        h = torch.relu(self.gin2(h, adj))
        graph_emb = h.mean(dim=0)
        return self.head(graph_emb).squeeze(-1)


def train_and_eval_gnn(train_graphs, y_train, test_graphs, y_test, epochs: int = 45, lr: float = 0.01, seed: int = 0):
    torch.manual_seed(seed)
    model = GraphLevelGINRegressor(in_dim=4, hidden_dim=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for (adj, x), y in zip(train_graphs, y_train):
            pred = model(torch.from_numpy(x), torch.from_numpy(adj))
            total_loss = total_loss + loss_fn(pred, torch.tensor(y))
        total_loss = total_loss / len(train_graphs)
        total_loss.backward()
        optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for adj, x in test_graphs:
            pred = model(torch.from_numpy(x), torch.from_numpy(adj))
            preds.append(pred.item())
    preds = np.array(preds)
    return float(np.sqrt(np.mean((preds - y_test) ** 2)))


def eval_sinkhorn_kernel_ridge(train_graphs, y_train, test_graphs, y_test):
    # Precompute pairwise Wasserstein distances ONCE (exploiting symmetry)
    D_full = compute_pairwise_wasserstein(train_graphs, train_graphs)
    D_test = compute_pairwise_wasserstein(test_graphs, train_graphs)

    best_rmse = float("inf")
    n_val = max(4, int(0.2 * len(train_graphs)))
    D_sub_tr = D_full[:-n_val, :-n_val]
    D_sub_val = D_full[-n_val:, :-n_val]
    y_sub_train, y_sub_val = y_train[:-n_val], y_train[-n_val:]

    for gamma in [0.1, 0.3, 0.6]:
        K_tr = np.exp(-gamma * D_sub_tr)
        K_val = np.exp(-gamma * D_sub_val)
        for alpha in [1e-2, 1e-1, 0.5]:
            krr = ExplicitKernelRidge(alpha=alpha).fit(K_tr, y_sub_train)
            val_preds = krr.predict(K_val)
            val_rmse = float(np.sqrt(np.mean((val_preds - y_sub_val) ** 2)))
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_cfg = (gamma, alpha)

    gamma_opt, alpha_opt = best_cfg
    K_full = np.exp(-gamma_opt * D_full)
    K_test = np.exp(-gamma_opt * D_test)
    krr = ExplicitKernelRidge(alpha=alpha_opt).fit(K_full, y_train)
    preds = krr.predict(K_test)
    return float(np.sqrt(np.mean((preds - y_test) ** 2)))

def run_all_experiments():
    print("=" * 72)
    print("EXPERIMENT 1: Finite-Sample Crossover (Sinkhorn-WL Kernel vs Explicit GIN)")
    print("=" * 72)
    sample_sizes = [30, 60, 120, 240]
    n_seeds = 5  # Increase to 20 for full overnight run
    kernel_rmses_mean, kernel_rmses_std = [], []
    gnn_rmses_mean, gnn_rmses_std = [], []

    for n_train in sample_sizes:
        k_scores, g_scores = [], []
        for s in range(n_seeds):
            graphs, y, _ = generate_synthetic_sbm_dataset(n_train + 50, seed=100 * s + n_train)
            tr_g, te_g = graphs[:n_train], graphs[n_train:]
            tr_y, te_y = y[:n_train], y[n_train:]

            k_rmse = eval_sinkhorn_kernel_ridge(tr_g, tr_y, te_g, te_y)
            g_rmse = train_and_eval_gnn(tr_g, tr_y, te_g, te_y, epochs=40, seed=s)
            k_scores.append(k_rmse)
            g_scores.append(g_rmse)

        kernel_rmses_mean.append(np.mean(k_scores))
        kernel_rmses_std.append(np.std(k_scores))
        gnn_rmses_mean.append(np.mean(g_scores))
        gnn_rmses_std.append(np.std(g_scores))
        print(
            f"N_train = {n_train:3d} | Sinkhorn-KRR RMSE: {np.mean(k_scores):.4f} ± {np.std(k_scores):.4f} "
            f"| Explicit-GIN RMSE: {np.mean(g_scores):.4f} ± {np.std(g_scores):.4f}"
        )

    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Random Split vs Size-Grouped Distribution Shift Split")
    print("=" * 72)
    graphs, y, sizes = generate_synthetic_sbm_dataset(180, min_nodes=10, max_nodes=32, seed=2026)

    # 1. Naive Random Split
    idx = np.random.default_rng(42).permutation(len(graphs))
    tr_idx, te_idx = idx[:130], idx[130:]
    gnn_random_rmse = train_and_eval_gnn(
        [graphs[i] for i in tr_idx], y[tr_idx], [graphs[i] for i in te_idx], y[te_idx], epochs=45, seed=42
    )

    # 2. Strict Structural Size Split (Train on smaller motifs, evaluate OOD on largest graphs)
    sorted_idx = np.argsort(sizes)
    tr_struct, te_struct = sorted_idx[:130], sorted_idx[130:]
    gnn_struct_rmse = train_and_eval_gnn(
        [graphs[i] for i in tr_struct], y[tr_struct], [graphs[i] for i in te_struct], y[te_struct], epochs=45, seed=42
    )
    krr_struct_rmse = eval_sinkhorn_kernel_ridge(
        [graphs[i] for i in tr_struct], y[tr_struct], [graphs[i] for i in te_struct], y[te_struct]
    )

    degradation_pct = ((gnn_struct_rmse - gnn_random_rmse) / gnn_random_rmse) * 100.0
    print(f"GIN Random Split Test RMSE        : {gnn_random_rmse:.4f}")
    print(f"GIN Size-Grouped Split Test RMSE  : {gnn_struct_rmse:.4f} (+{degradation_pct:.1f}% OOD degradation)")
    print(f"Sinkhorn-KRR Size-Grouped RMSE    : {krr_struct_rmse:.4f}")

    # Plot both experiments for the GitHub README
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    # Panel A: Crossover Curve
    axes[0].errorbar(sample_sizes, kernel_rmses_mean, yerr=kernel_rmses_std, marker="o", linewidth=2, capsize=4, label="Sinkhorn-WL Kernel Ridge")
    axes[0].errorbar(sample_sizes, gnn_rmses_mean, yerr=gnn_rmses_std, marker="s", linewidth=2, capsize=4, label="2-Layer Explicit GIN")
    axes[0].set_title("A. Finite-Sample Performance Crossover", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Training Sample Size (N graphs)")
    axes[0].set_ylabel("Test RMSE (Lower is Better)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel B: Dirichlet Energy Decay across depth (Oversmoothing vs Residual Spectral Control)
    torch.manual_seed(7)
    adj_t = torch.from_numpy(graphs[0][0])
    x_t = torch.from_numpy(graphs[0][1])
    depths = list(range(1, 11))
    vanilla_energy, residual_energy = [], []
    h_van, h_res = x_t.clone(), x_t.clone()
    layers = [ExplicitGCNLayer(4, 4) for _ in depths]
    with torch.no_grad():
        for lyr in layers:
            h_van = torch.tanh(lyr(h_van, adj_t))
            h_van = h_van / (torch.norm(h_van, dim=-1, keepdim=True) + 1e-8)
            vanilla_energy.append(dirichlet_energy(h_van, adj_t).item())

            # Spectral/Dirichlet residual preservation (Oct 2026 ongoing experiment)
            h_res = 0.82 * torch.tanh(lyr(h_res, adj_t)) + 0.18 * x_t
            h_res = h_res / (torch.norm(h_res, dim=-1, keepdim=True) + 1e-8)
            residual_energy.append(dirichlet_energy(h_res, adj_t).item())

    axes[1].plot(depths, vanilla_energy, marker="o", linewidth=2, label="Unregularized Message Passing")
    axes[1].plot(depths, residual_energy, marker="^", linewidth=2, linestyle="--", label="Dirichlet-Preserving Residual (Oct 2026)")
    axes[1].set_yscale("log")
    axes[1].set_title("B. Layer-Wise Dirichlet Energy E(X) = Tr(X^T L X)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Message Passing Depth (Layers)")
    axes[1].set_ylabel("Normalized Dirichlet Energy (Log Scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/results_oct2026.png")
    print("\nSaved benchmark figure to figures/results_oct2026.png")


if __name__ == "__main__":
    run_all_experiments()