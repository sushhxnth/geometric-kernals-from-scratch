import numpy as np
from scipy.spatial.distance import cdist


def compute_structural_node_embeddings(adj: np.ndarray, x: np.ndarray, steps: int = 2) -> np.ndarray:
    """
    Continuous Weisfeiler-Lehman (WL) diffusion embeddings:
    Concatenates [X, S X, S^2 X, ..., S^k X] where S is normalized adjacency.
    """
    n = adj.shape[0]
    a_tilde = adj + np.eye(n)
    deg = a_tilde.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    s_op = d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]

    features = [x]
    curr = x
    for _ in range(steps):
        curr = s_op @ curr
        features.append(curr)
    return np.concatenate(features, axis=1)


def sinkhorn_log_domain(cost_matrix: np.ndarray, reg: float = 0.2, max_iter: int = 15, tol: float = 1e-3) -> float:
    """
    Fast stabilized Sinkhorn-Knopp scaling for entropic Optimal Transport distance.
    """
    n, m = cost_matrix.shape
    mu = np.full(n, 1.0 / n)
    nu = np.full(m, 1.0 / m)

    # Stabilized Gibbs kernel
    c_norm = cost_matrix / (np.max(cost_matrix) + 1e-8)
    K = np.exp(-c_norm / reg) + 1e-12

    u = np.ones(n)
    v = np.ones(m)
    for _ in range(max_iter):
        u = mu / (K @ v)
        v = nu / (K.T @ u)

    pi = u[:, None] * K * v[None, :]
    return float(np.sum(pi * cost_matrix))


def compute_pairwise_wasserstein(graphs_A, graphs_B, wl_steps: int = 2, sinkhorn_reg: float = 0.2) -> np.ndarray:
    """
    Computes the pairwise Sinkhorn-Wasserstein distance matrix W(G_i, G_j) once.
    If graphs_A is graphs_B, exploits symmetry W(G_i, G_j) == W(G_j, G_i) to halve runtime.
    """
    emb_A = [compute_structural_node_embeddings(adj, x, steps=wl_steps) for adj, x in graphs_A]
    symmetric = graphs_A is graphs_B
    emb_B = emb_A if symmetric else [compute_structural_node_embeddings(adj, x, steps=wl_steps) for adj, x in graphs_B]

    dist_mat = np.zeros((len(emb_A), len(emb_B)))
    for i, ea in enumerate(emb_A):
        start_j = i + 1 if symmetric else 0
        for j in range(start_j, len(emb_B)):
            cost = cdist(ea, emb_B[j], metric="sqeuclidean")
            d = sinkhorn_log_domain(cost, reg=sinkhorn_reg)
            dist_mat[i, j] = d
            if symmetric:
                dist_mat[j, i] = d
    return dist_mat


def compute_sinkhorn_gram_matrix(graphs_A, graphs_B, wl_steps: int = 2, sinkhorn_reg: float = 0.2, gamma: float = 0.5) -> np.ndarray:
    dist_mat = compute_pairwise_wasserstein(graphs_A, graphs_B, wl_steps=wl_steps, sinkhorn_reg=sinkhorn_reg)
    return np.exp(-gamma * dist_mat)


class ExplicitKernelRidge:
    """
    Closed-form Kernel Ridge Regression: alpha = (K + lambda * I)^{-1} y
    """
    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.dual_coef_ = None

    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        n = K_train.shape[0]
        reg_matrix = K_train + self.alpha * np.eye(n)
        self.dual_coef_ = np.linalg.solve(reg_matrix, y_train)
        return self

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        return K_test_train @ self.dual_coef_