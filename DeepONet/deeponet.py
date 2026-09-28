"""DeepONet: a neural network that learns an OPERATOR (Lu et al., 2021).

An operator G maps a whole input function u to a whole output function G(u).
DeepONet approximates the value of G(u) at any point y of the output domain:

    G_theta(u)(y) = sum_{k=1}^p  b_k(u) * t_k(y)  +  b0
                                 \\____/   \\____/
                                 branch    trunk

  branch net : (u(x_1), ..., u(x_m)) -> (b_1, ..., b_p)   coefficients
  trunk net  : y                     -> (t_1, ..., t_p)   basis functions
  b0         : one trainable scalar bias

* The branch sees the input function ONLY through its values at m fixed
  "sensor" points x_1..x_m. The sensors are the same for every sample.
* The trunk sees ONLY the query coordinate y (e.g. y = x, or y = (x, t)).
  It never sees u.
* p is the number of basis functions: all information about u has to pass
  through these p numbers.

Link to ROM/pod.py, where u(x, t) ~= mean + sum_k a_k(t) phi_k(x):

    trunk output  t_k(y)  <->  POD mode phi_k(x)   (a basis, learned here)
    branch output b_k(u)  <->  coefficient a_k      (from a network here,
                                                     not from a reduced ODE)
    bias b0               <->  the mean (a scalar here, not a field)

Data layout used in this folder ("aligned" data): every input function is
evaluated at the SAME Q query points. Then one trunk pass serves the whole
batch and the forward pass is a single matrix product:

    U : (N, m)   N input functions sampled at m sensors
    Y : (Q, d)   Q query points in d dimensions
    B = branch(U)            (N, p)
    T = trunk(Y)             (Q, p)
    S = B @ T^T + b0         (N, Q)   S[i, j] = G(u_i)(y_j)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network. sizes = [n_in, hidden_1, ..., n_out].

    activate_last=False leaves the output layer linear (no activation).
    """

    def __init__(self, sizes, activation=nn.Tanh, activate_last=False):
        super().__init__()
        layers = []
        n_layers = len(sizes) - 1
        for i in range(n_layers):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.xavier_normal_(linear.weight)  # standard choice for tanh
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < n_layers - 1 or activate_last:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """Plain ("vanilla") DeepONet with an MLP branch and an MLP trunk.

    Parameters
    ----------
    branch_sizes : [m, hidden..., p]   m = number of sensors
    trunk_sizes  : [d, hidden..., p]   d = dimension of the query point y

    Two details that matter more than they look:
    * The branch's last layer is LINEAR: coefficients b_k can take any sign
      and size, like POD coefficients.
    * The trunk's last layer KEEPS its activation: the t_k are basis
      functions, and a nonlinear last layer makes them richer. (With a linear
      last layer, every t_k would be a linear combination of the same
      hidden features, which weakens the basis.)
    """

    def __init__(self, branch_sizes, trunk_sizes):
        super().__init__()
        if branch_sizes[-1] != trunk_sizes[-1]:
            raise ValueError("branch and trunk must end with the same width p")
        self.p = branch_sizes[-1]
        self.branch = MLP(branch_sizes, activate_last=False)
        self.trunk = MLP(trunk_sizes, activate_last=True)
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, U, Y):
        """U: (N, m) sensor values, Y: (Q, d) query points -> (N, Q)."""
        B = self.branch(U)  # (N, p) coefficients, one row per input function
        T = self.trunk(Y)   # (Q, p) basis functions, one row per query point
        return B @ T.T + self.b0

    def basis(self, Y):
        """The p learned basis functions evaluated at Y: (Q, p)."""
        return self.trunk(Y)

    def coefficients(self, U):
        """The p coefficients of each input function: (N, p)."""
        return self.branch(U)
