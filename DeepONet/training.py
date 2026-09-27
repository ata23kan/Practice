"""Generic DeepONet training loop -- independent of the operator being learned.

Loss: mean relative L2 error over the functions in a mini-batch,

    loss = (1/N_batch) sum_i  || G_theta(u_i) - G(u_i) ||_2 / || G(u_i) ||_2

where the norms are over the Q query points. The relative (not absolute)
error keeps small-amplitude functions from being ignored by the optimizer.

Mini-batches are drawn over INPUT FUNCTIONS; each batch uses all Q query
points (aligned data, see deeponet.py). Generalization is always measured on
held-out input functions -- the real question for an operator is "does it
work for a u it has never seen?", not "does it work at a new y?".
"""

import numpy as np
import torch


def to_tensor(*arrays):
    return [torch.as_tensor(a, dtype=torch.float32) for a in arrays]


def relative_l2(pred, true):
    """Per-function relative L2 error: (N, Q), (N, Q) -> (N,)."""
    return torch.linalg.norm(pred - true, dim=1) / torch.linalg.norm(true, dim=1)


def train(model, U_train, S_train, Y, U_test, S_test,
          n_iters=10000, batch_size=100, lr=1e-3, lr_decay=0.5,
          decay_every=2000, log_every=500, seed=0):
    """Train with Adam, halving the learning rate every `decay_every` steps.

    All data arguments are torch tensors: U (N, m), S (N, Q), Y (Q, d).
    Returns a history dict with the logged iterations and errors.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_every, lr_decay)

    history = {"iter": [], "train": [], "test": []}
    n_train = U_train.shape[0]
    for it in range(1, n_iters + 1):
        idx = torch.as_tensor(rng.choice(n_train, batch_size, replace=False))
        loss = relative_l2(model(U_train[idx], Y), S_train[idx]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % log_every == 0 or it == 1:
            train_err = evaluate(model, U_train, S_train, Y).mean()
            test_err = evaluate(model, U_test, S_test, Y).mean()
            history["iter"].append(it)
            history["train"].append(train_err)
            history["test"].append(test_err)
            print(f"iter {it:6d}  loss {loss.item():.3e}  "
                  f"train {train_err:.3e}  test {test_err:.3e}")
    return history


def evaluate(model, U, S, Y):
    """Relative L2 error of every function in (U, S), as a numpy array (N,)."""
    with torch.no_grad():
        return relative_l2(model(U, Y), S).numpy()


def predict(model, U, Y):
    with torch.no_grad():
        return model(U, Y).numpy()
