"""Forecasting models matching JOURNAL Sections 2.4 and 2.8.

Four architectures, all trained on the same target and loss:

* :class:`Gated` -- the multiscale gated network (MSGN) and its increments-only ablation.
* :class:`MLP` -- a single-hidden-layer dense baseline on the 45 features.
* :class:`LSTM` -- a sequential baseline on the raw 60-day return path.

Parameter counts are asserted by ``tests/test_model_architectures.py`` and quoted in the
manuscript: 923 (MSGN), 353 (LogSig-Inc), 1,505 (MLP), 4,513 (LSTM).
"""

import torch
from torch import nn

BLOCK = 15


class Gated(nn.Module):
    """Multiscale gated network (MSGN): one branch per window, softmax-weighted.

    Each window's coordinates pass through their own ReLU branch, a softmax gate
    computed from all features weights the three scales, and a linear head returns the
    forecast (JOURNAL Eq. 2). With ``block = 15`` the network has 923 parameters; with
    ``block = 5`` -- the increments-only ablation -- it has 353.

    Parameters
    ----------
    block
        Features per window: 15 for a full order-2 log-signature, 5 for increments only.
    hidden
        Width of each branch.
    """

    def __init__(self, block: int = BLOCK, hidden: int = 16) -> None:
        super().__init__()
        self.block = block
        self.branch_w5 = nn.Linear(block, hidden)
        self.branch_w20 = nn.Linear(block, hidden)
        self.branch_w60 = nn.Linear(block, hidden)
        self.gate = nn.Linear(block * 3, 3)
        self.head = nn.Linear(hidden, 1)

    @property
    def branches(self) -> tuple[nn.Linear, ...]:
        """The three per-window branches, ordered shortest window first."""
        return (self.branch_w5, self.branch_w20, self.branch_w60)

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax gate activations of shape ``(..., 3)`` for the three time scales."""
        return torch.softmax(self.gate(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from a ``(..., block * 3)`` feature tensor."""
        chunks = x.split(self.block, -1)
        if len(chunks) != len(self.branches):
            raise ValueError(f"expected {len(self.branches)} feature blocks, got {len(chunks)}")
        h = torch.stack([torch.relu(b(z)) for b, z in zip(self.branches, chunks, strict=True)], -2)
        return self.head((self.gates(x).unsqueeze(-1) * h).sum(-2)).squeeze(-1)


class MLP(nn.Module):
    """Single-hidden-layer dense baseline on the 45 features (1,505 parameters).

    Parameters
    ----------
    n_in
        Input width.
    hidden
        Hidden width.
    """

    def __init__(self, n_in: int = 45, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from a ``(..., n_in)`` feature tensor."""
        return self.head(torch.relu(self.fc1(x))).squeeze(-1)


class LSTM(nn.Module):
    """Single-layer LSTM baseline on the raw 60-day return path (4,513 parameters).

    Parameters
    ----------
    hidden
        Hidden width.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from a ``(batch, n_assets, 60, 1)`` sequence tensor."""
        b, n, t, c = x.shape
        _, (h, _) = self.lstm(x.reshape(b * n, t, c))
        return self.head(h[-1]).reshape(b, n)


class MSGN(Gated):
    """The manuscript's primary architecture: 923 parameters over 45 features."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__(block=15, hidden=hidden_dim)


class LogSigInc(Gated):
    """Increments-only ablation: 353 parameters over the 15 linear path increments."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__(block=5, hidden=hidden_dim)


class MLPBaseline(MLP):
    """Named alias of :class:`MLP` with the manuscript's hyperparameters."""

    def __init__(self, in_features: int = 45, hidden_dim: int = 32) -> None:
        super().__init__(n_in=in_features, hidden=hidden_dim)


class LSTMBaseline(LSTM):
    """Named alias of :class:`LSTM` with the manuscript's hyperparameters."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32) -> None:
        super().__init__(hidden=hidden_dim)


def build_model(name: str, hidden: int = 16) -> nn.Module:
    """Instantiate a model by its configuration key.

    Parameters
    ----------
    name
        One of ``"logsig"``, ``"logsig_inc"``, ``"mlp"`` or ``"lstm"``.
    hidden
        Hidden width for the gated architectures; ignored by the baselines, which use
        the widths reported in the manuscript.

    Returns
    -------
    nn.Module
        An untrained model.

    Raises
    ------
    KeyError
        If ``name`` is not one of the four registered models.
    """
    match name:
        case "logsig":
            return Gated(BLOCK, hidden)
        case "logsig_inc":
            return Gated(5, hidden)
        case "mlp":
            return MLP(45, 32)
        case "lstm":
            return LSTM(32)
    raise KeyError(name)


def _zscore(a: torch.Tensor) -> torch.Tensor:
    """Cross-sectional z-score along the asset axis, clipped to ``[-3, 3]``."""
    return torch.clamp((a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-8), -3.0, 3.0)


def composite_loss(pred: torch.Tensor, y: torch.Tensor, var_y: float, lam: float = 1.0) -> torch.Tensor:
    """NMSE plus a differentiable cross-sectional ranking surrogate (JOURNAL Eq. 3).

    Parameters
    ----------
    pred, y
        ``(n_dates, n_assets)`` forecasts and targets.
    var_y
        Training variance of the target, used to normalise the squared error.
    lam
        Weight on the ``1 - rho`` ranking term.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    nmse = ((pred - y) ** 2).mean() / var_y
    u, v = _zscore(pred), _zscore(y)
    u, v = u - u.mean(1, keepdim=True), v - v.mean(1, keepdim=True)
    rho = (u * v).sum(1) / (((u**2).sum(1) + 1e-8).sqrt() * ((v**2).sum(1) + 1e-8).sqrt())
    return nmse + lam * (1.0 - rho).mean()
