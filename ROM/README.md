# ROM: POD + Galerkin reduced-order models, a minimal starting point

A plain projection-based reduced-order model (ROM) in NumPy/SciPy, with two
PDEs and one "where it breaks" experiment. The POD and Galerkin code is
generic and knows nothing about the physics; each PDE only supplies a
full-order model. Every file is short and explains its own math at the top.
Read them in the order below.

## The idea

Discretize a PDE in space and you get a large system of ODEs, the
**full-order model (FOM)**:

$$
\frac{d\mathbf{u}}{dt} \;=\; A\,\mathbf{u} + N(\mathbf{u}, t) + \mathbf{f}(t),
\qquad \mathbf{u}(t) \in \mathbb{R}^{N}.
$$

- $A \in \mathbb{R}^{N \times N}$ is the linear part (here: diffusion),
- $N(\mathbf{u}, t)$ is an optional nonlinear part (here: Burgers convection),
- $\mathbf{f}(t)$ is an optional external forcing.

$N$ is the number of grid points (a few hundred here, millions in real CFD).
In practice the solution does not use all $N$ directions: it moves on a
low-dimensional subspace. A ROM finds that subspace from data and solves the
equations only inside it:

$$
\mathbf{u}(t) \;\approx\; \bar{\mathbf{u}} + \sum_{k=1}^{r} a_k(t)\, \boldsymbol{\phi}_k
\;=\; \bar{\mathbf{u}} + \Phi\, \mathbf{a}(t),
\qquad \Phi \in \mathbb{R}^{N \times r}, \quad \Phi^{\top}\Phi = I_r, \quad r \ll N.
$$

Two steps, and they are completely separate:

1. **POD** (offline, from data): find the mean $\bar{\mathbf{u}}$ and the
   basis $\Phi$.
2. **Galerkin projection** (online, from the equations): find an ODE for the
   $r$ coefficients $\mathbf{a}(t)$ and integrate it.

POD alone is just data compression. Galerkin is what makes it a *model*:
the coefficients come from the governing equations, not from a fit.

### Step 1: Proper Orthogonal Decomposition (POD)

Run the FOM once and store $M$ snapshots as the columns of a matrix:

$$
U = \big[\, \mathbf{u}(t_1) \;\; \mathbf{u}(t_2) \;\; \cdots \;\; \mathbf{u}(t_M) \,\big]
\in \mathbb{R}^{N \times M}.
$$

Subtract the time mean (a Reynolds-style decomposition into mean +
fluctuations) and take the thin SVD of the fluctuations:

$$
\bar{\mathbf{u}} = \frac{1}{M}\sum_{j=1}^{M} \mathbf{u}(t_j),
\qquad
U - \bar{\mathbf{u}}\,\mathbf{1}^{\top} \;=\; \Phi\, \Sigma\, \Psi^{\top}.
$$

The columns of $\Phi$ are the **POD modes**, ordered by the singular values
$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. By the Eckart–Young theorem, the
first $r$ modes give the best rank-$r$ approximation of the snapshots in the
least-squares sense. No other $r$-dimensional basis does better on this data.

**Choosing $r$.** $\sigma_k^2$ is the "energy" in mode $k$. We keep the
smallest $r$ whose modes hold a fraction $\tau$ of the total energy:

$$
E(r) \;=\; \frac{\sum_{k=1}^{r} \sigma_k^2}{\sum_{k} \sigma_k^2} \;\ge\; \tau,
\qquad \tau = 0.9999 \text{ by default.}
$$

The faster the singular values decay, the smaller $r$ can be. The singular
value plot (`*_singular_values.png`) is the first thing to look at for any
new problem.

**The covariance view (PCA) and the method of snapshots.** The same modes are
the eigenvectors of the spatial covariance matrix, since

$$
C = U_f\, U_f^{\top} = \Phi\, \Sigma^2\, \Phi^{\top} \in \mathbb{R}^{N \times N},
\qquad U_f = U - \bar{\mathbf{u}}\,\mathbf{1}^{\top}.
$$

When $N > M$ (more grid points than snapshots, the usual case) the
$N \times N$ matrix is wasteful. Sirovich's **method of snapshots**
eigendecomposes the small $M \times M$ temporal correlation matrix instead and
maps back:

$$
K = U_f^{\top} U_f = \Psi\, \Sigma^2\, \Psi^{\top} \in \mathbb{R}^{M \times M},
\qquad
\boldsymbol{\phi}_k = \frac{1}{\sigma_k}\, U_f\, \boldsymbol{\psi}_k .
$$

`pod.py` implements both routes (`method="svd"` and `method="covariance"`).
They give the same singular values, and the same modes up to a sign. SVD is
the numerically safer default: forming $U_f^{\top} U_f$ squares the condition
number, so the smallest singular values lose accuracy.

### Step 2: Galerkin projection

Insert the ansatz $\mathbf{u} \approx \bar{\mathbf{u}} + \Phi\mathbf{a}$ into
the FOM. It will not be satisfied exactly, so there is a residual
$\mathbf{R}(\mathbf{a}, t)$. The **Galerkin condition** asks the residual to be
orthogonal to the subspace we are working in, $\Phi^{\top}\mathbf{R} = 0$.
Using $\Phi^{\top}\Phi = I$ this gives an $r$-dimensional ODE:

$$
\frac{d\mathbf{a}}{dt} \;=\;
\underbrace{\Phi^{\top} A\, \Phi}_{A_r}\, \mathbf{a}
\;+\; \underbrace{\Phi^{\top} A\, \bar{\mathbf{u}}}_{\text{offset}}
\;+\; \Phi^{\top} N\!\left(\bar{\mathbf{u}} + \Phi\mathbf{a},\, t\right)
\;+\; \Phi^{\top} \mathbf{f}(t).
$$

- $A_r \in \mathbb{R}^{r \times r}$ and the offset $\in \mathbb{R}^{r}$ are
  computed once, offline.
- The **offset** only exists because we subtracted the mean. Without mean
  subtraction $\bar{\mathbf{u}} = 0$ and it vanishes.
- The **initial condition** must be projected consistently:

$$
\mathbf{a}(0) = \Phi^{\top}\big(\mathbf{u}(0) - \bar{\mathbf{u}}\big).
$$

  This is *not* zero in general, even when $\mathbf{u}(0) = 0$. Forgetting it
  gives a ROM that starts from the wrong state.

After integrating, lift back to full space:
$\mathbf{u}_{\text{ROM}}(t) = \bar{\mathbf{u}} + \Phi\,\mathbf{a}(t)$.

### Linear vs nonlinear: where the speed-up goes

For a **linear** problem, every RHS evaluation costs $O(r^2)$ instead of
$O(N^2)$ (or $O(N)$ for a sparse $A$). This is the real win.

For a **nonlinear** term, this code is "naive" Galerkin. At each RHS call it

1. lifts $\mathbf{a}$ to the full state $\mathbf{u} = \bar{\mathbf{u}} + \Phi\mathbf{a}$ (cost $O(Nr)$),
2. evaluates $N(\mathbf{u}, t)$ at full order (cost $O(N)$),
3. projects back with $\Phi^{\top}$ (cost $O(Nr)$).

This is mathematically correct Galerkin projection, but the cost still scales
with $N$. **Hyper-reduction** methods (DEIM, EIM, gappy POD) fix this by
evaluating the nonlinear term at only a few chosen points. They are not
implemented here. For the quadratic Burgers term one can also precompute a
reduced tensor $\Phi^{\top} N(\Phi\,\cdot)$ of size $r^3$.

### Error measure

The FOM and ROM are compared snapshot by snapshot with the relative $L^2$
error over space:

$$
e(t) \;=\; \frac{\big\lVert \mathbf{u}_{\text{FOM}}(t) - \mathbf{u}_{\text{ROM}}(t) \big\rVert_2}
               {\big\lVert \mathbf{u}_{\text{FOM}}(t) \big\rVert_2}.
$$

Note that all runs here are **reproductive**: the ROM is tested on the same
trajectory whose snapshots built the basis. This checks the Galerkin
dynamics, not generalization. Predicting a new parameter or a new initial
condition is the harder (and more useful) question; see "Things to try".

### Link to DeepONet

A DeepONet (see `DeepONet/`) has the same "basis times coefficients"
structure:

| POD ROM | DeepONet |
|---|---|
| POD modes $\boldsymbol{\phi}_k(x)$ | trunk basis functions $t_k(y)$ |
| modal coefficients $a_k(t)$ | branch coefficients $b_k(u)$ |
| mean field $\bar{u}(x)$ | bias $b_0$ (a scalar) |

The difference: the ROM gets the basis from an SVD and the coefficients by
solving a reduced ODE derived from the PDE. A DeepONet *learns* both from
input–output data, with no equations involved.

## The three examples

**1. Heat equation with a moving source** (linear)

$$
\partial_t u = \nu\, \partial_{xx} u + f(x,t), \qquad x \in (0,1), \qquad
u(0,t) = u(1,t) = 0, \qquad u(x,0) = 0,
$$

$$
f(x,t) = 5\, \exp\!\left( -\frac{(x - x_0(t))^2}{2\,(0.05)^2} \right),
\qquad x_0(t) = 0.5 + 0.3 \sin\!\left(\frac{2\pi t}{4}\right).
$$

$\nu = 0.01$, $t \in [0, 8]$, 199 interior points, 400 snapshots, second-order
central differences, BDF time stepping. The source sweeps back and forth, so
the solution has a clear non-zero mean with fluctuations on top: a good case
to see why mean subtraction and the offset term matter. The forcing is
projected as $\Phi^{\top}\mathbf{f}(t)$ and the reduced system is linear, so
the exact Jacobian $A_r$ is passed to the stiff solver.

**2. Viscous Burgers equation** (nonlinear)

$$
\partial_t u + \partial_x\!\left( \frac{u^2}{2} \right) = \nu\, \partial_{xx} u,
\qquad x \in [0, 2) \text{ periodic},
$$

$$
u(x,0) = \sin(\pi x) + 0.5 \sin(2\pi x).
$$

$\nu = 0.02$, $t \in [0, 1.5]$, 400 grid points, 300 snapshots. The flux
$u^2/2$ is differentiated with central differences (fine because $\nu$ keeps
the front resolved; no shock-capturing needed). The initial wave steepens
into a front that is then smoothed by viscosity. POD is computed with the
method of snapshots here ($N = 400 > M = 300$, so the $300 \times 300$ matrix
is used) and cross-checked against the direct SVD.

**3. Where naive Galerkin breaks: low-viscosity Burgers**

Example 2 picks $r$ by the energy threshold, so it quietly adds modes as the
problem gets harder and never shows failure. Here $r = 6$ is **fixed** and
$\nu$ is lowered from $0.02$ to $10^{-4}$ (800 grid points). As fronts get
steeper:

- the singular values decay more slowly, so 6 modes hold less energy;
- the error grows steadily;
- the ROM **overshoots** the true amplitude, $\max|u_{\text{ROM}}| > \max|u_{\text{FOM}}|$.

The overshoot is the classic instability of Galerkin ROMs for
advection-dominated flows. In the FOM, energy cascades from large scales to
small ones and is dissipated there. The truncated modes are exactly those
small scales, so the ROM has nowhere to send the energy and it piles up in
the resolved modes. A second figure shows the same $\nu = 0.0005$ case with
$r = 6, 15, 40$: more modes help, but slowly. This gap is what **closure
models** (eddy-viscosity closures, data-driven or physics-informed
corrections acting on $\mathbf{a}$) are designed to fill.

## Files

| file | what it does |
|---|---|
| `pod.py` | POD via SVD or covariance / method of snapshots; energy $E(r)$ and rank selection |
| `galerkin_rom.py` | generic Galerkin ROM: $A_r$, mean offset, reduced IC, integration, reconstruction |
| `plotting.py` | shared plots (singular values, FOM vs ROM snapshots, error vs time) and the relative $L^2$ error |
| `fom_heat1d.py` | FOM 1: heat equation with a moving Gaussian source (Dirichlet) |
| `fom_burgers1d.py` | FOM 2: viscous Burgers (periodic); exposes $N(\mathbf{u})$ for reuse in the ROM |
| `run_heat_galerkin.py` | example 1: linear Galerkin ROM with forcing and mean subtraction |
| `run_burgers_galerkin.py` | example 2: nonlinear (naive) Galerkin ROM, POD via method of snapshots |
| `run_burgers_low_viscosity.py` | example 3: fixed $r$, decreasing $\nu$, showing the ROM failure mode |

Suggested reading order: `pod.py` → `galerkin_rom.py` → `fom_heat1d.py` +
`run_heat_galerkin.py` → `fom_burgers1d.py` + `run_burgers_galerkin.py` →
`run_burgers_low_viscosity.py`.

The key design point: the ROM code never imports a FOM. A FOM only hands over
$A$, and optionally $N(\mathbf{u}, t)$ and $\mathbf{f}(t)$ as plain Python
functions. Adding a new PDE means writing a new `fom_*.py` and a short run
script; `pod.py` and `galerkin_rom.py` stay unchanged.

## Running

Requirements: `numpy`, `scipy`, `matplotlib`.

```bash
cd ROM
python run_heat_galerkin.py           # ~2 s
python run_burgers_galerkin.py        # ~2 s
python run_burgers_low_viscosity.py   # ~70 s (8 FOM solves at 800 points)
```

Each FOM module can also be run on its own (`python fom_heat1d.py`) to print
the snapshot matrix size. Figures are written to `ROM/figures/`.

Typical output:

| example | $N \times M$ | chosen $r$ | energy | mean / max relative error |
|---|---|---|---|---|
| heat | $199 \times 400$ | 9 | 99.991% | 0.85% / 11% |
| Burgers, $\nu = 0.02$ | $400 \times 300$ | 7 | 99.996% | 0.31% / 0.91% |

The covariance and SVD singular values agree to about $10^{-10}$. The heat
maximum error is at the very start of the run ($t \approx 0.02$), where
$\mathbf{u}$ is still close to zero and a relative error is dominated by its
small denominator. The median error over the run is about 0.5%.

For the low-viscosity sweep at fixed $r = 6$:

| $\nu$ | 0.02 | 0.005 | 0.001 | 0.0001 |
|---|---|---|---|---|
| energy in 6 modes | 99.98% | 99.37% | 97.95% | 97.20% |
| mean relative error | 0.7% | 6.5% | 18% | 25% |
| amplitude overshoot | 1.01× | 1.14× | 1.47× | 1.32× |

Notice that "97% energy" sounds good but gives a 25% error: an energy
threshold that works for diffusion is far too loose for sharp fronts.

## Things to try

Each of these changes one or two lines and teaches one idea.

1. **Change $r$ by hand** in `run_heat_galerkin.py` (replace `choose_rank`).
   Plot the mean error against $r$ and compare it with the singular value
   plot. The two curves should decay together.
2. **Turn off mean subtraction**: `compute_pod(U, subtract_mean=False)`. The
   offset term vanishes, and the first mode has to represent the mean. How
   many extra modes do you need for the same error?
3. **Drop the offset or the projected IC** in a run script (set `offset` or
   `a0` to zeros). Both bugs are easy to make and produce a wrong ROM that
   still runs without error.
4. **Compare the two POD routes** in `run_heat_galerkin.py`
   (`method="covariance"`). Here $N < M$, so the $N \times N$ spatial matrix
   is used instead. Look at how the smallest singular values differ from
   the SVD ones.
5. **Predict, don't reproduce**: build the basis from snapshots on
   $t \in [0, 4]$ only and integrate the ROM to $t = 8$. Or build it at one
   $\nu$ and run the ROM at another (the ROM's $A$ changes, the basis does
   not). This is the real test of a ROM.
6. **Time it**: measure the wall-clock time of the FOM solve vs the ROM
   integration for heat and for Burgers. The linear ROM should be much
   faster; the naive nonlinear one much less so, because it still evaluates
   $N(\mathbf{u})$ at full order.
7. **Remove the cost of the nonlinear term**: for Burgers, precompute the
   reduced quadratic tensor, or implement DEIM, so the RHS no longer touches
   $N$ grid points.
8. **Add a closure**: in example 3, add a simple eddy-viscosity term
   $-\nu_c\, \mathrm{diag}(k^2)\,\mathbf{a}$ (more damping on higher modes)
   to the reduced RHS and see how much of the overshoot it removes.

## References

- L. Sirovich, *Turbulence and the dynamics of coherent structures, Part I:
  Coherent structures*, Quarterly of Applied Mathematics 45(3), 561–571 (1987).
  (The method of snapshots.)
- P. Holmes, J. L. Lumley, G. Berkooz, C. W. Rowley, *Turbulence, Coherent
  Structures, Dynamical Systems and Symmetry*, 2nd ed., Cambridge University
  Press (2012).
- S. Chaturantabut, D. C. Sorensen, *Nonlinear model reduction via discrete
  empirical interpolation*, SIAM Journal on Scientific Computing 32(5),
  2737–2764 (2010). (DEIM hyper-reduction.)
- J. S. Hesthaven, G. Rozza, B. Stamm, *Certified Reduced Basis Methods for
  Parametrized Partial Differential Equations*, Springer (2016).
