# DeepONet: a minimal starting point

A plain DeepONet in PyTorch, with three small examples. Every file is short
and explains its own math at the top. Read them in the order below.

## The idea

An **operator** maps a whole function to another whole function. We want to
learn an operator $\mathcal{G}$ that takes an input function $u$ and returns
an output function $\mathcal{G}(u)$, which we can evaluate at any point $y$:

$$
\mathcal{G} : u \;\mapsto\; \mathcal{G}(u), \qquad \mathcal{G}(u) : y \;\mapsto\; \mathcal{G}(u)(y).
$$

A DeepONet approximates it as a sum of $p$ products:

$$
\mathcal{G}(u)(y) \;\approx\; \mathcal{G}_\theta(u)(y) \;=\; \sum_{k=1}^{p} b_k(u)\, t_k(y) \;+\; b_0 .
$$

- **Branch net**: reads the input function at $m$ fixed sensor points and
  returns $p$ **coefficients**:

$$
\big(u(x_1), \dots, u(x_m)\big) \;\mapsto\; \big(b_1(u), \dots, b_p(u)\big).
$$

- **Trunk net**: reads a query point $y$ and returns $p$ **basis functions**:

$$
y \;\mapsto\; \big(t_1(y), \dots, t_p(y)\big).
$$

- $b_0$ is one trainable scalar bias.

The branch only sees $u$ and the trunk only sees $y$. Everything the network
knows about $u$ has to pass through the $p$ numbers $b_k(u)$.

### Link to POD

A POD reduced-order model (see `ROM/`) writes

$$
u(x,t) \;\approx\; \bar{u}(x) + \sum_{k=1}^{r} a_k(t)\, \phi_k(x).
$$

It has the same structure:

| DeepONet | POD ROM |
|---|---|
| trunk basis functions $t_k(y)$ | POD modes $\phi_k(x)$ |
| branch coefficients $b_k(u)$ | modal coefficients $a_k(t)$ |
| bias $b_0$ (a scalar) | mean field $\bar{u}(x)$ |

The difference is that a DeepONet *learns* both the basis and the
coefficients. A ROM gets the basis from an SVD and the coefficients by
solving a reduced ODE.

### How the code computes it (matrix form)

Take $N$ input functions, each sampled at the same $m$ sensors, and $Q$ query
points in $d$ dimensions:

$$
U \in \mathbb{R}^{N \times m}, \qquad Y \in \mathbb{R}^{Q \times d}.
$$

One branch pass and one trunk pass give

$$
B = \mathrm{branch}(U) \in \mathbb{R}^{N \times p}, \qquad
T = \mathrm{trunk}(Y) \in \mathbb{R}^{Q \times p},
$$

and the whole prediction is a single matrix product:

$$
S \;\approx\; B\, T^{\top} + b_0, \qquad S_{ij} = \mathcal{G}(u_i)(y_j).
$$

### Training

The loss is the mean relative $L^2$ error over the functions in a batch, with
the norm taken over the $Q$ query points:

$$
\mathcal{L}(\theta) \;=\; \frac{1}{N} \sum_{i=1}^{N}
\frac{\big\lVert \mathcal{G}_\theta(u_i) - \mathcal{G}(u_i) \big\rVert_2}
     {\big\lVert \mathcal{G}(u_i) \big\rVert_2}.
$$

The error is always reported on **input functions that were not used in
training**. For an operator, the question that matters is "does it work for a
new $u$?", not "does it work at a new $y$?".

### Random input functions

An operator is learned over a *distribution* of input functions. We draw
them from a Gaussian random field with the kernel

$$
k(x, x') \;=\; \exp\!\left( -\frac{(x - x')^2}{2 \ell^2} \right).
$$

The length scale $\ell$ sets how smooth the functions are: a smaller $\ell$
gives rougher functions, which are harder to learn. Samples are drawn with
the Karhunen–Loève expansion. If $K = V \Lambda V^{\top}$ is the covariance
matrix on the grid, then

$$
u = V \Lambda^{1/2} z, \qquad z \sim \mathcal{N}(0, I).
$$

## The three operators

**1. Antiderivative** (linear; the "hello world" of operator learning)

$$
\mathcal{G}(u)(y) \;=\; \int_0^{y} u(s)\, ds, \qquad y \in [0, 1].
$$

**2. Heat equation** (linear; time goes into the trunk)

$$
\partial_t u = \nu\, \partial_{xx} u, \qquad x \in (0,1), \qquad
u(0,t) = u(1,t) = 0, \qquad u(x,0) = u_0(x).
$$

The initial conditions are random sine series, so the exact solution is known:

$$
u_0(x) = \sum_{k=1}^{K} c_k \sin(k\pi x), \quad c_k \sim \frac{\mathcal{N}(0,1)}{k}
\qquad \Longrightarrow \qquad
u(x,t) = \sum_{k=1}^{K} c_k\, e^{-\nu (k\pi)^2 t} \sin(k\pi x).
$$

The operator is $\mathcal{G} : u_0 \mapsto u(x,t)$ with trunk input
$y = (x, t)$. Every solution lies in the span of $K$ functions, so the output
data has rank exactly $K$ ($K = 10$ here). You can see this as a cliff in the
singular value plot.

**3. Viscous Burgers equation** (nonlinear)

$$
\partial_t u + \partial_x\!\left( \frac{u^2}{2} \right) = \nu\, \partial_{xx} u,
\qquad x \in [0, 1) \text{ periodic}, \qquad u(x,0) = u_0(x).
$$

The operator is $\mathcal{G} : u_0 \mapsto u(\cdot, T)$ with $\nu = 0.02$ and
$T = 0.3$. The data comes from a Fourier pseudo-spectral solver.

## Files

| file | what it does |
|---|---|
| `deeponet.py` | the architecture: `MLP` and `DeepONet` (branch, trunk, bias) |
| `grf.py` | random input functions (Gaussian random fields) |
| `training.py` | training loop, relative $L^2$ loss, evaluation |
| `plotting.py` | shared plots |
| `data_antiderivative.py` | operator 1: $u \mapsto \int_0^y u(s)\,ds$ |
| `data_heat1d.py` | operator 2: heat equation, $u_0 \mapsto u(x,t)$ (exact solution) |
| `data_burgers1d.py` | operator 3: viscous Burgers, $u_0 \mapsto u(\cdot,T)$ (spectral solver) |
| `run_antiderivative.py` | example 1: the minimal workflow, plus the learned trunk basis |
| `run_heat.py` | example 2: 2D trunk input $y = (x,t)$, output rank vs $p$ |
| `run_burgers.py` | example 3: a nonlinear operator |

Suggested reading order: `deeponet.py` → `grf.py` → `training.py` →
`data_antiderivative.py` + `run_antiderivative.py` → heat → Burgers.

## Running

Requirements: `numpy`, `scipy`, `matplotlib`, `torch`.

```bash
cd DeepONet
python run_antiderivative.py   # ~20 s
python run_heat.py             # ~50 s
python run_burgers.py          # ~30 s
```

Figures are written to `DeepONet/figures/`. Typical relative $L^2$ test
errors (on input functions not seen in training): about 1% (antiderivative),
2% (heat) and 8% (Burgers). The Burgers number is not a bug. It is what a
plain DeepONet gives on a nonlinear problem with fronts, and it is the
starting point for every improvement in the literature.

## Things to try

Each of these changes one or two lines and teaches one idea.

1. **Change $p$** in any run script. How small can it be before the error
   jumps? Compare with the singular value plot (`*_output_spectrum.png`).
2. **Rougher inputs**: lower `length_scale` ($\ell$) in `data_antiderivative.py`.
   Rough functions need more sensors and more data.
3. **Spectral bias**: set `n_modes = 20` in `run_heat.py`. The operator is still
   linear, but the trunk now has to draw $\sin(20\pi x)$ and training gets much
   slower.
4. **Harder physics**: set `NU = 0.01` in `run_burgers.py`. Steeper fronts make the
   operator harder to learn with the same data and network.
5. **Break the architecture on purpose**: in `deeponet.py`, make the trunk's
   last layer linear (`activate_last=False`) or remove the bias $b_0$. What
   happens?
6. **Freeze the trunk**: replace the trunk with the first $p$ POD modes of
   the training outputs (the SVD of `S_train`) and train only the branch.
   This is "POD-DeepONet", and it links directly to `ROM/`.

## Reference

L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, *Learning nonlinear
operators via DeepONet based on the universal approximation theorem of
operators*, Nature Machine Intelligence 3, 218–229 (2021).
