"""Generic FNO training loop -- independent of the operator being learned.

Loss: mean relative L2 error over the fields in a mini-batch,

    loss = (1/N_batch) sum_i  || G_theta(a_i) - u_i ||_2 / || u_i ||_2,

with the norm over all grid points. The relative error keeps small-amplitude
outputs (strongly decayed solutions at large diffusivity) from being ignored.

Data are torch tensors: inputs A (N, C_in, n, n), outputs U (N, C_out, n, n).
Everything is kept on one device (the GPU when available); the data sets in
this folder are a few tens of MB.
"""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_l2(pred, true):
    """Per-sample relative L2 error: (N, ...) -> (N,)."""
    diff = (pred - true).flatten(1)
    return torch.linalg.norm(diff, dim=1) / torch.linalg.norm(true.flatten(1), dim=1)


def train(model, A_train, U_train, A_test, U_test, epochs=150, batch_size=20,
          lr=1e-3, weight_decay=1e-4, log_every=10, seed=0):
    """Adam with a cosine learning-rate schedule. Returns a history dict."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = A_train.shape[0]
    n_batches = n_train // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * n_batches)

    history = {"epoch": [], "train": [], "test": []}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.as_tensor(rng.permutation(n_train), device=A_train.device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            loss = relative_l2(model(A_train[idx]), U_train[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            train_err = evaluate(model, A_train, U_train).mean()
            test_err = evaluate(model, A_test, U_test).mean()
            history["epoch"].append(epoch)
            history["train"].append(train_err)
            history["test"].append(test_err)
            print(f"epoch {epoch:4d}  loss {loss.item():.3e}  "
                  f"train {train_err:.3e}  test {test_err:.3e}")
    return history


@torch.no_grad()
def predict(model, A, batch_size=100):
    """Model output for all inputs, in chunks to bound GPU memory."""
    model.eval()
    return torch.cat([model(A[i:i + batch_size]) for i in range(0, A.shape[0], batch_size)])


def evaluate(model, A, U):
    """Relative L2 error of every sample, as a numpy array (N,)."""
    return relative_l2(predict(model, A), U).cpu().numpy()
