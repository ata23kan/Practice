# Fourier Neural Operator (FNO): a minimal starting point

A plain 2D FNO in PyTorch, with two heat-conduction examples that ask the
same question: **can we predict the solution for a diffusivity we did not
train on?** Every file is short and explains its own math at the top. Read
them in the order given under [Files](#files).

## The idea

Like DeepONet (see `DeepONet/`), an FNO learns an **operator**: a map from an
input function to an output function,

$$
\mathcal{G} : a \;\mapsto\; u = \mathcal{G}(a).
$$

Here "operator" means the **solution operator** of the PDE, not the
differential operator. Physical parameters (diffusivity, Reynolds number,
Rayleigh number), initial conditions, forcing, and coefficient fields are all
**inputs** of $\mathcal{G}$. Changing $\nu$ does not need a new network: it is
a new input to the same $\mathcal{G}$.

An FNO is a stack of layers acting on fields $v_l(x) \in \mathbb{R}^{d_v}$
($d_v$ = `width` channels at every grid point):

$$
v_0 = P\,a, \qquad
v_{l+1} = \sigma\big(W_l v_l + \mathcal{K}_l v_l + b_l\big), \qquad
u = Q\,v_L .
$$

- $P$, $W_l$ and $Q$ act **pointwise**: the same small matrix at every grid point.
- $\mathcal{K}_l$ is the only **non-local** part. It is a kernel integral operator,
  which generalizes the Green's-function solution of a linear PDE,
  $u(x) = \int G(x,y) f(y)\,dy$.

### The Fourier layer

Assume the kernel is translation invariant, $\kappa(x,y) = \kappa(x-y)$. The
integral is then a convolution, and the convolution theorem turns it into a
product in Fourier space:

$$
(\mathcal{K} v)(x) = \int \kappa(x-y)\, v(y)\, dy
= \mathcal{F}^{-1}\big(R \cdot \mathcal{F} v\big)(x),
\qquad R(k) \in \mathbb{C}^{d_v \times d_v}.
$$

$R$ is learned **directly in Fourier space**, and only the lowest
$k_{\max}$ = `modes` wavenumbers are kept ($R(k) = 0$ above them). The
consequences:

- One layer costs $O(n \log n)$ (the FFT) but couples every point to every other point.
- The parameters belong to **wavenumbers, not grid points**, so the same
  network runs on any grid that resolves those modes. This is
  *discretization invariance*.
- For linear, constant-coefficient PDEs the solution operator is a Fourier
  multiplier, which a single Fourier layer represents **exactly**. For heat:
  $\hat u(k,T) = e^{-\nu |k|^2 T}\, \hat u_0(k)$.
- The truncation removes high frequencies. The pointwise path $W$ and the
  nonlinearity $\sigma$ put them back (products create new frequencies:
  $e^{ik_1x}e^{ik_2x} = e^{i(k_1+k_2)x}$).

A Fourier layer is structured like one step of a **pseudo-spectral solver**:
the linear part is applied in Fourier space, the nonlinear part pointwise in
physical space.

### In 2D: which modes are kept

`rfft2` stores only $k_y \ge 0$, because a real field has Hermitian symmetry.
$k_x$ still runs over both signs, so the kept modes are two corner blocks of
the spectrum, each with its own weights:

$$
R \in \mathbb{C}^{2 \times k_{\max} \times k_{\max} \times d_v \times d_v}.
$$

### Translation equivariance

No grid coordinates are fed to the network. Every operation commutes with
periodic shifts, so the FNO is exactly translation equivariant:

$$
\mathcal{G}_\theta(\tau_s a) = \tau_s\, \mathcal{G}_\theta(a).
$$

Both heat problems below are periodic with no fixed position in space, so
this is the right bias. For walls or a fixed forcing $f(x)$, append $(x, y)$
as input channels.

## The two operators

Both examples solve heat conduction on the periodic square $[0, 2\pi)^2$ with
$T = 1$. Initial conditions come from a band-limited Gaussian random field
(`grf2d.py`).

### 1. Scalar diffusivity: $(u_0, \nu) \mapsto u(\cdot, T)$

$$
u_t = \nu\, \Delta u .
$$

The exact solution is $\hat u(k,T) = e^{-\nu |k|^2 T} \hat u_0(k)$, so there
is no solver and the data are exact. The operator is linear in $u_0$ and
nonlinear in $\nu$.

**How $\nu$ enters.** $\nu$ is broadcast as a constant input channel. A
constant field has only the $k = 0$ Fourier mode, and the spectral layer
mixes channels **linearly**, so $\nu$ cannot directly set how strongly mode
$k$ decays. The network has to learn roughly

$$
u(T) \approx \sum_c \alpha_c(\nu)\; \mathcal{F}^{-1}\!\big(e^{-\tau_c |k|^2}\,\hat u_0\big),
$$

i.e. a set of decay rates, mixed with $\nu$-dependent weights through the
pointwise nonlinearities. This means **interpolating in $\nu$**: it works
inside the training range and not outside it.

### 2. Diffusivity field: $(u_0, \kappa) \mapsto u(\cdot, T)$

$$
u_t = \nabla \cdot \big(\kappa(x, y)\, \nabla u\big),
\qquad \kappa = \bar\kappa\, e^{\sigma g(x,y)} .
$$

This models a heterogeneous material. The field $g$ is a smooth random field,
so $\kappa$ varies by a factor of about 16 across the domain, around a mean
level $\bar\kappa$. There is no closed form. The reference solver is
pseudo-spectral in space with RK4 in time and needs hundreds of steps.

**This is the problem FNO is built for:**

- The input is a whole field with rich Fourier content, so the spectral
  layers use it directly. (The scalar $\nu$ of problem 1 lives only in the
  $k = 0$ mode.)
- The map is nonlinear in $\kappa$.
- The classical solve is expensive, while the FNO needs one forward pass.

DeepONet would have to push the whole $\kappa$ field through a finite set of
sensors into $p$ coefficients. A POD-ROM would need a new Galerkin system for
every $\kappa$.

## Predicting for unseen diffusivities

Both examples train on a diffusivity range $[0.02, 0.08]$ and test from
$0.005$ (4× below the range) to $0.64$ (8× above it).

- **Inside the range**, the FNO predicts the solution for diffusivities it
  never saw. This is the useful regime: every new $\nu$ or $\kappa$ costs one
  forward pass.
- **Outside the range**, the error grows quickly. No data-driven surrogate
  knows what it was not shown. The same holds for a parametric POD-ROM, and
  for a PINN trained at one parameter value.

### Extrapolating with exact physics: the semigroup property

Diffusing with $c\kappa$ for time $T$ is the same as diffusing with $\kappa$
for time $cT$. The heat flow is also a semigroup, $S(t_1 + t_2) = S(t_1)\,S(t_2)$.
Together these give

$$
\mathcal{G}(u_0,\; c\,\kappa) = \underbrace{\mathcal{G}\big(\cdots \mathcal{G}(u_0,\; \tfrac{c}{m}\kappa) \cdots,\; \tfrac{c}{m}\kappa\big)}_{m \text{ times}} .
$$

So for a diffusivity **above** the training range, choose $m$ so that
$c/m$ falls inside the range, and apply the FNO $m$ times. This is exact
physics, not network extrapolation.

It only works in one direction: a diffusion cannot be "un-composed", so there
is no such trick below the range. There you need more training data, or a
physics-informed (PINO) fine-tune at the new value.

Repeated application feeds the network a smaller, decayed $u$. To keep that
input in-distribution we use another exact property: $u(T)$ is linear in
$u_0$, so

$$
\mathcal{G}(c\,u_0) = c\,\mathcal{G}(u_0).
$$

`FNO2d(homogeneous=True)` enforces this by scaling $u_0$ to unit RMS on the
way in and scaling back on the way out.

## Files

| file | what it does |
|---|---|
| `fno.py` | the architecture: `SpectralConv2d` (the Fourier layer) and `FNO2d` |
| `grf2d.py` | random 2D input fields: band-limited periodic GRF (a Fourier series with random coefficients) |
| `training.py` | training loop, relative $L^2$ loss, evaluation |
| `plotting.py` | shared plots, including the measured Fourier multiplier $H(k)$ |
| `data_heat2d.py` | operator 1: $(u_0, \nu) \mapsto u(T)$, exact solution |
| `data_heat2d_kappa.py` | operator 2: $(u_0, \kappa) \mapsto u(T)$, pseudo-spectral RK4 solver |
| `run_heat2d_nu.py` | example 1: unseen $\nu$, semigroup trick, resolution test, learned multiplier |
| `run_heat2d_kappa.py` | example 2: unseen $\bar\kappa$, semigroup trick, resolution test, speed-up |

Suggested reading order: `fno.py` → `grf2d.py` → `training.py` →
`data_heat2d.py` + `run_heat2d_nu.py` → `data_heat2d_kappa.py` + `run_heat2d_kappa.py`.

## Running

Requirements: `numpy`, `matplotlib`, `torch`. A GPU is used when available.

```bash
cd FNO
python run_heat2d_nu.py      # ~4 min on a laptop GPU (RTX A1000)
python run_heat2d_kappa.py   # ~5 min (1 min of it is data generation)
```

Figures are written to `FNO/figures/`.

## Results

These are mean relative $L^2$ test errors on unseen inputs, for an FNO with 4
layers, width 32, 12 modes, 1000 training samples on a $64^2$ grid and 150
epochs (about 3 min of training on an RTX A1000 laptop GPU).

| diffusivity (× training range) | scalar $\nu$, direct | scalar $\nu$, semigroup | field $\kappa$, direct | field $\kappa$, semigroup |
|---|---|---|---|---|
| 0.005 (4× below) | 15% | not applicable | 2.7% | not applicable |
| 0.0141 (1.4× below) | 2.4% | not applicable | 1.0% | not applicable |
| **0.02 to 0.08 (inside)** | **0.10 to 0.41%** | not needed | **0.79 to 1.4%** | not needed |
| 0.113 (1.4× above) | 3.4% | 0.22% | 2.4% | 1.2% |
| 0.32 (4× above) | 41% | 1.2% | 17% | 2.0% |
| 0.64 (8× above) | 92% | 3.2% | 62% | 3.6% |

Other checks:

- **Resolution invariance.** The model trained on $64^2$ gives the same error
  on $48^2$ and $128^2$ with no retraining (0.14% for $\nu$, 0.90% for
  $\kappa$).
- **Speed.** For 200 variable-$\kappa$ solves, the reference solver takes
  13.7 s and the FNO 0.08 s, about 180× faster. Most of the value of operator
  learning is this online speed-up, paid for by the offline data generation
  and training.

What the figures show:

- `heat2d_nu_error_vs_nu.png`: a "valley" over the training range. The
  semigroup curve brings large-$\nu$ extrapolation back to the percent level.
- `heat2d_nu_transfer.png`: the multiplier the FNO actually applies.
  - Inside the range, it follows $e^{-\nu |k|^2 T}$ over many decades up to the
    truncation at $k = 12$.
  - Above $k = 12$ it is **flat**. The spectral path is cut off there, and the
    pointwise path $W$ can only apply the same constant multiple to every high mode.
  - Outside the range, the learned curve simply has the wrong decay rate.
- `heat2d_kappa_fields.png`: $\kappa$, $u_0$, the solver solution, the FNO
  prediction and the error, below, inside and above the range.

Why the $\kappa$ problem extrapolates more gently below the range: the
*mean* level $\bar\kappa = 0.005$ is unseen, but *pointwise* diffusivity
values that small already occur inside the training fields, which vary by a
factor of about 16 around their mean. The FNO has seen that local physics,
just not as a domain average. With the scalar $\nu$, an unseen value is
unseen everywhere.

## Things to try

Each of these changes one or two lines and teaches one idea.

1. **Fewer modes.** Set `modes = 4` in a run script. The multiplier plot
   (`heat2d_nu_transfer.png`) shows what truncation does, and what the
   pointwise path $W$ can and cannot recover.
2. **One layer, no nonlinearity.** Train problem 1 at a *fixed* $\nu$ with
   `n_layers = 1`. The architecture then contains the exact answer, and
   $R(k)$ should approach $e^{-\nu |k|^2 T}$. Then let $\nu$ vary again. One
   layer is no longer enough, for the $k = 0$ reason explained above.
3. **A wider training range.** Train on $\nu \in [0.005, 0.64]$ and compare
   with the semigroup curve. Covering the range with data is always the first fix.
4. **Turn off `homogeneous`.** The in-range error barely changes, but the
   semigroup curve gets worse. Repeated application now feeds inputs of the
   wrong amplitude.
5. **Break equivariance.** Add $(\sin x, \cos x, \sin y, \cos y)$ as input
   channels. On these periodic problems it only costs data. It becomes
   necessary once a fixed forcing $f(x)$ is added.
6. **Compare with DeepONet** on problem 2, with $\kappa$ and $u_0$ sampled at
   sensors in the branch and $(x, y)$ in the trunk (see `DeepONet/`).
7. **PINO.** Below the training range, fine-tune the trained FNO with only
   the PDE residual at the new $\nu$. For problem 1, predict $u$ at several
   times so that $u_t$ is available, and compute $\Delta u$ spectrally
   ($\Delta \leftrightarrow -|k|^2$). This is a PINN solve warm-started by an operator.

## References

- Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart,
  A. Anandkumar, *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021, arXiv:2010.08895.
- N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart,
  A. Anandkumar, *Neural Operator: Learning Maps Between Function Spaces*,
  JMLR 24 (2023).
- N. Kovachki, S. Lanthaler, S. Mishra, *On universal approximation and
  error bounds for Fourier Neural Operators*, JMLR 22 (2021).
- Z. Li et al., *Physics-Informed Neural Operator for Learning Partial
  Differential Equations*, arXiv:2111.03794.
