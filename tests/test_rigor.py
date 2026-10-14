import torch
import pytest
from src.operators import ExplicitGCNLayer, ExplicitGINLayer, ExplicitGraphSAGE, dirichlet_energy


@pytest.mark.parametrize("LayerClass", [ExplicitGCNLayer, ExplicitGINLayer, ExplicitGraphSAGE])
def test_node_permutation_equivariance(LayerClass):
    """
    Verifies strict permutation equivariance:
    For any permutation matrix P in S_n, f(P X, P A P^T) == P f(X, A)
    """
    torch.manual_seed(42)
    n_nodes, in_dim, out_dim = 14, 8, 6
    x = torch.randn(n_nodes, in_dim, dtype=torch.float64)
    raw_adj = (torch.rand(n_nodes, n_nodes, dtype=torch.float64) > 0.65).to(torch.float64)
    adj = ((raw_adj + raw_adj.T) > 0).to(torch.float64)

    layer = LayerClass(in_dim, out_dim, dtype=torch.float64)
    layer.eval()

    # Random permutation matrix P
    perm = torch.randperm(n_nodes)
    P = torch.eye(n_nodes, dtype=torch.float64)[perm]

    out_original = layer(x, adj)
    out_permuted_inputs = layer(P @ x, P @ adj @ P.T)
    permuted_out = P @ out_original

    rel_err = torch.norm(out_permuted_inputs - permuted_out) / torch.norm(permuted_out)
    assert rel_err.item() < 1e-12, f"Equivariance violated! Relative error: {rel_err.item():.3e}"


@pytest.mark.parametrize("LayerClass", [ExplicitGCNLayer, ExplicitGINLayer, ExplicitGraphSAGE])
def test_float64_finite_difference_gradcheck(LayerClass):
    """
    Verifies analytical backward pass against centered finite-difference numerical
    gradients in float64 with relative error strictly under 1e-6.
    """
    torch.manual_seed(108)
    n_nodes, in_dim, out_dim = 7, 5, 4
    x = torch.randn(n_nodes, in_dim, dtype=torch.float64, requires_grad=True)
    raw_adj = torch.rand(n_nodes, n_nodes, dtype=torch.float64)
    adj = (0.5 * (raw_adj + raw_adj.T)).requires_grad_(True)

    layer = LayerClass(in_dim, out_dim, dtype=torch.float64)

    def func(x_in, adj_in):
        return layer(x_in, adj_in)

    passed = torch.autograd.gradcheck(
        func,
        (x, adj),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-6,
        raise_exception=True,
    )
    assert passed


def test_dirichlet_energy_permutation_invariance():
    """
    Verifies that Dirichlet energy E(X, A) = Tr(X^T L X) is strictly invariant to node relabeling.
    """
    torch.manual_seed(7)
    n_nodes, dim = 12, 5
    x = torch.randn(n_nodes, dim, dtype=torch.float64)
    raw_adj = (torch.rand(n_nodes, n_nodes, dtype=torch.float64) > 0.5).to(torch.float64)
    adj = ((raw_adj + raw_adj.T) > 0).to(torch.float64)

    perm = torch.randperm(n_nodes)
    P = torch.eye(n_nodes, dtype=torch.float64)[perm]

    e1 = dirichlet_energy(x, adj)
    e2 = dirichlet_energy(P @ x, P @ adj @ P.T)
    assert torch.allclose(e1, e2, atol=1e-12)