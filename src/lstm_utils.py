"""Shared infrastructure for the Arctic sea ice LSTM notebooks.

This module consolidates what used to be copy-pasted across ``04_basic_lstm``,
``05_multivariate_lstm`` and ``06_seq2seq_lstm`` into one tested implementation:

* a single :class:`SequenceDataset` covering univariate, multivariate, lagged,
  cyclical-time and multi-step (seq2seq) setups;
* a single :class:`IceExtentLSTM` whose output width equals the forecast horizon;
* a :func:`train_model` loop with a **proper train/validation/test split**
  (validation is held out from the training era — the test set is never touched
  during training), fixed seeds, gradient clipping, LR scheduling, early
  stopping and optional CUDA mixed precision (AMP);
* :func:`save_checkpoint` / :func:`load_checkpoint`, which persist a
  self-contained bundle (weights **plus** the normalization scaler, feature
  list, config and split dates) so predictions can be denormalized and
  evaluated later without re-deriving anything.

Design notes
------------
The old notebooks used the 2020-2023 *test* set for early stopping, LR
scheduling and checkpoint selection, which leaks the test set into model
selection. Here validation comes from within the training era (see
:func:`temporal_split`) and the test set is only used for final evaluation.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

Scaler = Tuple[np.ndarray, np.ndarray]  # (mean, std), each shape (1, n_features)


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (incl. CUDA) for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trade a little speed for determinism; safe to relax on the GPU box.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(verbose: bool = True) -> torch.device:
    """Return the best available device, printing GPU details when present."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using device: cuda ({name}, {mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using device: cpu")
    return device


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparameters and training settings for one LSTM run.

    Defaults are sized for a laptop CPU smoke test; bump ``hidden_size``,
    ``batch_size`` and ``num_epochs`` (and set ``amp=True``) on the GPU box.
    """

    sequence_length: int = 30
    forecast_horizon: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    num_epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 15          # early-stopping patience (epochs)
    lr_patience: int = 5        # ReduceLROnPlateau patience
    lr_factor: float = 0.5
    seed: int = 42
    amp: bool = False           # CUDA mixed precision; ignored on CPU
    num_workers: int = 0
    pin_memory: bool = False


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame,
    train_years: range,
    val_years: range,
    test_years: range,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dated frame into train/val/test by calendar year (no overlap).

    The validation years must sit inside the training era and the test years
    must be strictly later, so nothing from the test period influences model
    selection.
    """
    years = pd.to_datetime(df[date_col]).dt.year

    def _slice(yr: range) -> pd.DataFrame:
        return df[years.isin(set(yr))].sort_values(date_col).reset_index(drop=True)

    parts = {"train": (_slice(train_years), train_years),
             "val": (_slice(val_years), val_years),
             "test": (_slice(test_years), test_years)}
    for name, (part, yr) in parts.items():
        if part.empty:
            raise ValueError(f"{name} split is empty for years {list(yr)}")
    return parts["train"][0], parts["val"][0], parts["test"][0]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SequenceDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for single- or multi-step ice extent forecasting.

    Args:
        data: DataFrame with a ``date`` column plus the requested feature columns.
        sequence_length: Number of past timesteps fed to the model.
        forecast_horizon: Number of future steps to predict (1 = single step).
        features: Feature columns to use; defaults to ``[target]``.
        target: Column being forecast.
        scaler: ``(mean, std)`` from the training set; if None, computed here
            (only do this on the training split to avoid leakage).
        lag_features: Optional ``{column: [lag_days, ...]}`` extra lagged inputs.
        add_cyclical_time: Append sin/cos day-of-year features.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        features: Optional[Sequence[str]] = None,
        target: str = "extent_mkm2",
        scaler: Optional[Scaler] = None,
        lag_features: Optional[Dict[str, Sequence[int]]] = None,
        add_cyclical_time: bool = False,
    ):
        df = data.sort_values("date").reset_index(drop=True).copy()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target = target

        self.features: List[str] = list(features) if features is not None else [target]

        if add_cyclical_time:
            doy = pd.to_datetime(df["date"]).dt.dayofyear
            df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
            df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
            self.features += ["day_of_year_sin", "day_of_year_cos"]

        if lag_features:
            for column, lags in lag_features.items():
                for lag in lags:
                    name = f"{column}_lag{lag}"
                    df[name] = df[column].shift(lag)
                    self.features.append(name)
            df = df.dropna().reset_index(drop=True)

        if target not in self.features:
            raise ValueError(f"target '{target}' must be among features {self.features}")
        self.target_idx = self.features.index(target)

        values = df[self.features].values.astype(np.float32)

        if scaler is None:
            mean = values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        else:
            self.mean, self.std = scaler

        self.data = (values - self.mean) / self.std
        self.dates = pd.to_datetime(df["date"]).reset_index(drop=True)

    @property
    def scaler(self) -> Scaler:
        return (self.mean, self.std)

    @property
    def target_scaler(self) -> Tuple[float, float]:
        """Scalar ``(mean, std)`` for the target column, for denormalizing y."""
        return float(self.mean[0, self.target_idx]), float(self.std[0, self.target_idx])

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx: int):
        s, h = self.sequence_length, self.forecast_horizon
        X = self.data[idx : idx + s]
        y = self.data[idx + s : idx + s + h, self.target_idx]  # shape (h,)
        return torch.from_numpy(X), torch.from_numpy(y.copy())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class IceExtentLSTM(nn.Module):
    """LSTM whose output width equals the forecast horizon.

    A horizon of 1 reproduces the single-step models from notebooks 04/05; a
    larger horizon reproduces the seq2seq model from notebook 06.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, seq, feat) -> (B, horizon)
        out, _ = self.lstm(x)          # nn.LSTM initializes hidden state to zeros
        return self.fc(out[:, -1, :])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, grad_clip=1.0):
    """Run one epoch; train if ``optimizer`` is given, else evaluate."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(X)
                loss = criterion(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    bundle_extra: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train with validation-based early stopping; return the loss history.

    Validation loss (never the test set) drives LR scheduling, early stopping
    and best-checkpoint selection. When ``checkpoint_path`` is given, the best
    model is saved as a self-contained bundle via :func:`save_checkpoint`.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.lr_patience, factor=config.lr_factor
    )
    amp_on = bool(config.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_on)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    since_best = 0

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Training {n_params:,} params | AMP={'on' if amp_on else 'off'} | "
              f"batch={config.batch_size} | max {config.num_epochs} epochs")

    for epoch in range(1, config.num_epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, device,
                        optimizer=optimizer, scaler=grad_scaler, grad_clip=config.grad_clip)
        va = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{config.num_epochs} | train {tr:.6f} | val {va:.6f}")

        if va < best_val:
            best_val = va
            since_best = 0
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, config, bundle_extra)
        else:
            since_best += 1
            if since_best >= config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val {best_val:.6f})")
                break

    history["best_val_loss"] = best_val
    history["epochs_trained"] = len(history["train_loss"])
    if verbose:
        print(f"Done. Best validation loss: {best_val:.6f}")
    return history


# --------------------------------------------------------------------------- #
# Checkpoints (self-contained bundles)
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    model: nn.Module,
    config: TrainConfig,
    extra: Optional[dict] = None,
) -> None:
    """Save weights plus everything needed to reuse the model.

    ``extra`` should carry the scaler and feature metadata, e.g.::

        {"scaler": (mean, std), "features": [...], "target": "extent_mkm2",
         "input_size": 7, "split": {"train": [1989, 2014], ...}}
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "model_class": type(model).__name__,
    }
    if extra:
        # Store scaler as plain arrays for portability.
        if "scaler" in extra:
            mean, std = extra["scaler"]
            extra = {**extra, "scaler": (np.asarray(mean), np.asarray(std))}
        bundle.update(extra)
    torch.save(bundle, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
    build_model: bool = True,
) -> dict:
    """Load a checkpoint bundle; optionally rebuild the model from config.

    Returns the bundle dict. When ``build_model`` is True, adds a ready-to-use
    ``model`` (in eval mode) reconstructed from the saved config and input size.
    """
    device = device or get_device(verbose=False)
    bundle = torch.load(path, map_location=device, weights_only=False)

    if build_model:
        cfg = TrainConfig(**bundle["config"])
        input_size = bundle.get("input_size")
        if input_size is None:
            raise ValueError("checkpoint missing 'input_size'; cannot rebuild model")
        model = IceExtentLSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            forecast_horizon=cfg.forecast_horizon,
            dropout=cfg.dropout,
        )
        model.load_state_dict(bundle["state_dict"])
        model.to(device).eval()
        bundle["model"] = model
        bundle["config_obj"] = cfg
    return bundle


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Return ``(preds, actuals)`` as numpy arrays in normalized units.

    Shapes are ``(N, horizon)``; for single-step models that is ``(N, 1)``.
    """
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            preds.append(model(X).cpu().numpy())
            actuals.append(y.numpy())
    return np.concatenate(preds), np.concatenate(actuals)


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert z-score normalization: ``x * std + mean`` (real units, Mkm²)."""
    return np.asarray(values) * std + mean


def make_loader(dataset, config: TrainConfig, shuffle: bool) -> torch.utils.data.DataLoader:
    """Build a DataLoader honoring the config's batch/worker/pin settings."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
