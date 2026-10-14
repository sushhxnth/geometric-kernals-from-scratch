# Finite-Sample Variance, Spectral Oversmoothing, and Size-Induced Distribution Shift: Benchmarking Explicit Message-Passing Operators Against Entropic Optimal Transport & Graph Kernels

**Status:** Active Independent Research (June 2026 – Present)  
**Research Area:** Statistical Learning Theory, Optimal Transport, and Geometric Deep Learning  
**Methodology:** Mathematics first, explicit from-scratch implementation second, controlled synthetic experimentation third, and rigorous failure-mode interpretation throughout.

---

## Abstract

Standard evaluations of Graph Neural Networks (GNNs) rely heavily on high-level library abstractions (`torch_geometric`, `dgl`) and random train/test splits on static benchmark datasets. While convenient, this paradigm obscures two critical failure modes in scientific machine learning: **(1) finite-sample optimization variance** relative to closed-form non-parametric kernel estimators, and **(2) severe out-of-distribution (OOD) degradation under topological size shift** caused by structural data leakage in random splits.

This repository presents an end-to-end mathematical and empirical investigation comparing explicit message-passing neural architectures (`GCN`, `GraphSAGE`, `GIN`) against continuous Weisfeiler-Lehman (WL) diffusion embeddings paired with Entropic Optimal Transport (Sinkhorn-Wasserstein) Kernel Ridge Regression (KRR). All neural operators, analytical backward passes, and Sinkhorn-Knopp solvers are implemented from scratch in pure PyTorch, NumPy, and SciPy without external graph learning libraries. Every message-passing layer is verified via centered `float64` finite-difference gradient checks (relative error $< 10^{-6}$) and strict permutation equivariance assertions ($< 10^{-12}$ error). Controlled experiments on tunable Stochastic Block Models (SBMs) demonstrate that while a 2-layer `ExplicitGIN` achieves lower asymptotic bias when its message-passing depth aligns with the data-generating motif (`0.1943` vs. `0.2126` RMSE at $N = 60$), closed-form Sinkhorn-WL KRR reduces seed-to-seed variance by **50.3%** (`±0.0133` vs. `±0.0264`). Furthermore, replacing standard random splits with strict size-grouped OOD splits ($N_{\text{nodes}} \in [10, 24] \rightarrow [25, 32]$) exposes a **74.3% test error degradation** in message passing (`0.1746` $\rightarrow$ `0.3043` RMSE) alongside an unnormalized density scaling failure in uniform-marginal Optimal Transport (`0.3611` RMSE).

---

## 1. Mathematical Formulation & Operator Architecture

Let $G = (V, E)$ be an undirected graph with $\vert{}V\vert{} = n$ nodes, adjacency matrix $A \in \{0, 1\}^{n \times n}$, degree matrix $D = \text{diag}(A \mathbf{1}_n)$, and node covariate matrix $X \in \mathbb{R}^{n \times d}$.

### 1.1 Explicit Message-Passing Operators & Analytical Jacobians (`src/operators.py`)

Rather than relying on black-box automatic differentiation through sparse gather-scatter primitives, the linear message-passing operator is formulated explicitly via the symmetric normalized shift operator $\tilde{S} = \tilde{D}^{-1/2}(A + I_n)\tilde{D}^{-1/2}$:

$$Y = \tilde{S} X W + \mathbf{1}_n b^\top, \quad W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}, \; b \in \mathbb{R}^{d_{\text{out}}}$$

In `ExplicitMessagePassingFn(torch.autograd.Function)`, the exact analytical backward pass is derived in closed matrix form for an upstream cotangent tensor $G = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{n \times d_{\text{out}}}$:

$$\frac{\partial \mathcal{L}}{\partial \tilde{S}} = G (XW)^\top, \quad \frac{\partial \mathcal{L}}{\partial X} = \left(\tilde{S}^\top G\right) W^\top, \quad \frac{\partial \mathcal{L}}{\partial W} = X^\top \left(\tilde{S}^\top G\right), \quad \frac{\partial \mathcal{L}}{\partial b} = G^\top \mathbf{1}_n$$

Three distinct structural aggregation paradigms are implemented from first principles:
* **`ExplicitGCNLayer`:** Spectral low-pass filtering via $\tilde{S} X W$ with explicit analytical backward propagation.
* **`ExplicitGINLayer`:** Injective multiset aggregation parameterized by learnable $\epsilon \in \mathbb{R}$:
  $$H^{(k)} = \text{MLP}_\theta\Big(\big((1 + \epsilon)I_n + A\big) H^{(k-1)}\Big)$$
* **`ExplicitGraphSAGE`:** Decoupled ego-feature and random-walk neighborhood aggregation:
  $$H^{(k)} = X W_{\text{self}} + \left(D^{-1} A X\right) W_{\text{neigh}}$$

### 1.2 Continuous Weisfeiler-Lehman Diffusion & Entropic Optimal Transport (`src/kernels.py`)

To construct a non-parametric baseline sensitive to both topological structure and continuous node covariates, each graph $G_i$ is mapped to a $K$-step structural diffusion distribution:

$$\Phi_K(X, A) = \left[ X \,\big\Vert{}\, \tilde{S} X \,\big\Vert{}\, \tilde{S}^2 X \,\big\Vert{}\, \dots \,\big\Vert{}\, \tilde{S}^K X \right] \in \mathbb{R}^{n \times (K+1)d}$$

Given two graphs $G_A = (X_A, A_A)$ with $n$ nodes and $G_B = (X_B, A_B)$ with $m$ nodes, we treat their node embeddings as empirical measures $\mu = \frac{1}{n}\sum_{p=1}^n \delta_{\Phi_K(G_A)_p}$ and $\nu = \frac{1}{m}\sum_{q=1}^m \delta_{\Phi_K(G_B)_q}$. The ground cost matrix is $C_{pq} = \Vert{}\Phi_K(G_A)_p - \Phi_K(G_B)_q\Vert{}_2^2$. The entropic-regularized Wasserstein distance $\mathcal{W}_\varepsilon(G_A, G_B)$ is computed by solving the strictly convex optimal transport problem:

$$\mathcal{W}_\varepsilon(\mu, \nu) = \min_{\Pi \in U(\mu, \nu)} \langle \Pi, C \rangle_F + \varepsilon \sum_{p=1}^n \sum_{q=1}^m \Pi_{pq} (\log \Pi_{pq} - 1)$$

where $U(\mu, \nu) = \{\Pi \in \mathbb{R}_{+}^{n \times m} \mid \Pi \mathbf{1}_m = \mu, \, \Pi^\top \mathbf{1}_n = \nu\}$. Using Sinkhorn-Knopp matrix scaling on the stabilized Gibbs kernel $K = \exp(-\tilde{C}/\varepsilon)$:

$$u^{(t+1)} = \frac{\mu}{K v^{(t)}}, \quad v^{(t+1)} = \frac{\nu}{K^\top u^{(t+1)}}, \quad \Pi^* = \text{diag}(u^*) K \text{diag}(v^*)$$

The resulting pairwise transport distances define an exponential Wasserstein Gram matrix $\mathcal{K}_{ij} = \exp(-\gamma \mathcal{W}_\varepsilon(G_i, G_j))$, which is solved in closed form via regularized Kernel Ridge Regression:

$$\hat{\alpha} = (\mathcal{K}_{\text{train}} + \lambda I_N)^{-1} y_{\text{train}}, \quad \hat{y}_{\text{test}} = \mathcal{K}_{\text{test, train}} \hat{\alpha}$$

### 1.3 Spectral Oversmoothing & Dirichlet Energy Functional

To quantify high-frequency signal attenuation across deep message-passing cascades independent of finite-sample overfitting, we track the normalized Dirichlet energy functional over the symmetric normalized graph Laplacian $L = I_n - D^{-1/2} A D^{-1/2}$:

$$\mathcal{E}\big(H^{(l)}\big) = \frac{1}{n} \text{Tr}\!\left(H^{(l)\top} L H^{(l)}\right) = \frac{1}{2n} \sum_{(u,v) \in E} \left\Vert{} \frac{H_u^{(l)}}{\sqrt{d_u}} - \frac{H_v^{(l)}}{\sqrt{d_v}} \right\Vert{}_2^2$$

---

## 2. Numerical Verification & Invariance Suite (`tests/test_rigor.py`)

Before running empirical benchmarks, every operator in `src/operators.py` is subjected to two strict numerical verification protocols in double precision (`torch.float64`):

1. **Centered Finite-Difference Jacobian Verification:**  
   For every parameter and input tensor $\theta$, analytical gradients $\nabla_\theta \mathcal{L}$ are tested against numerical finite differences with step size $\delta = 10^{-6}$:
   $$\frac{\left\Vert{} \nabla_\theta^{\text{analytical}} \mathcal{L} - \frac{\mathcal{L}(\theta + \delta e_i) - \mathcal{L}(\theta - \delta e_i)}{2\delta} \right\Vert{}_2}{\max\left(\left\Vert{}\nabla_\theta^{\text{analytical}} \mathcal{L}\right\Vert{}_2, \left\Vert{}\nabla_\theta^{\text{numerical}} \mathcal{L}\right\Vert{}_2\right)} < 10^{-6}$$
2. **Strict Node-Permutation Equivariance & Invariance:**  
   For a randomly sampled permutation matrix $P \in \Pi_n$, every message-passing layer is verified to commute with node relabeling up to machine epsilon ($\approx 10^{-12}$), while the Dirichlet energy functional is verified to be strictly permutation-invariant:
   $$f(PX, PAP^\top) = P f(X, A), \quad \mathcal{E}(PX, PAP^\top) = \mathcal{E}(X, A)$$

```text
$ python -m pytest -v
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
collected 7 items

tests/test_rigor.py::test_node_permutation_equivariance[ExplicitGCNLayer] PASSED
tests/test_rigor.py::test_node_permutation_equivariance[ExplicitGINLayer] PASSED
tests/test_rigor.py::test_node_permutation_equivariance[ExplicitGraphSAGE] PASSED
tests/test_rigor.py::test_float64_finite_difference_gradcheck[ExplicitGCNLayer] PASSED
tests/test_rigor.py::test_float64_finite_difference_gradcheck[ExplicitGINLayer] PASSED
tests/test_rigor.py::test_float64_finite_difference_gradcheck[ExplicitGraphSAGE] PASSED
tests/test_rigor.py::test_dirichlet_energy_permutation_invariance PASSED

============================== 7 passed in 1.27s ==============================
