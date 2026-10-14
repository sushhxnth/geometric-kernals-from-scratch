import math
import torch
import torch.nn as nn


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Computes symmetric normalized adjacency: \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}
    Differentiable with respect to edge weights in adj.
    """
    if add_self_loops:
        eye = torch.eye(adj.size(-1), dtype=adj.dtype, device=adj.device)
        adj = adj + eye
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)


def normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Computes symmetric normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}"""
    eye = torch.eye(adj.size(-1), dtype=adj.dtype, device=adj.device)
    adj_norm = normalize_adjacency(adj, add_self_loops=False)
    return eye - adj_norm


def dirichlet_energy(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Computes Dirichlet energy E(X) = 1/N * Tr(X^T L X) to quantify oversmoothing.
    """
    lap = normalized_laplacian(adj)
    energy = torch.trace(x.transpose(-1, -2) @ lap @ x) / x.size(0)
    return energy


class ExplicitMessagePassingFn(torch.autograd.Function):
    """
    Custom autograd function for explicit graph message passing: Y = S @ X @ W + b
    Writing the exact analytical Jacobian backward pass avoids black-box reliance
    and enables exact float64 finite-difference verification.
    """
    @staticmethod
    def forward(ctx, shift_op: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        support = x @ weight
        out = shift_op @ support
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(shift_op, x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift_op, x, weight, bias = ctx.saved_tensors
        # dL/dS = (dL/dY) @ (X @ W)^T
        grad_shift = grad_output @ (x @ weight).transpose(-1, -2)
        # Propagate gradient across graph shift operator: M = S^T @ (dL/dY)
        propagated_grad = shift_op.transpose(-1, -2) @ grad_output
        # dL/dX = M @ W^T
        grad_x = propagated_grad @ weight.transpose(-1, -2)
        # dL/dW = X^T @ M
        grad_weight = x.transpose(-1, -2) @ propagated_grad
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_shift, grad_x, grad_weight, grad_bias


class ExplicitGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        shift_op = normalize_adjacency(adj, add_self_loops=True)
        return ExplicitMessagePassingFn.apply(shift_op, x, self.weight, self.bias)


class ExplicitGINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) operator:
    H^{(k)} = MLP((1 + eps) * H^{(k-1)} + A @ H^{(k-1)})
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 0.0, dtype=torch.float32):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor([eps], dtype=dtype))
        self.fc1 = nn.Linear(in_features, out_features, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.size(-1), dtype=adj.dtype, device=adj.device)
        shift_op = adj + (1.0 + self.eps) * eye
        agg = shift_op @ x
        return self.fc2(self.act(self.fc1(agg)))


class ExplicitGraphSAGE(nn.Module):
    """
    Explicit GraphSAGE mean-aggregation operator:
    H^{(k)} = W_self X + W_neigh (D^{-1} A X)
    """
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=True, dtype=dtype)
        self.w_neigh = nn.Linear(in_features, out_features, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        neigh_mean = (adj / deg) @ x
        return self.w_self(x) + self.w_neigh(neigh_mean)