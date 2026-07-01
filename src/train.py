"""Headless, reproducible LSTM training — one command per run.

This is the CLI companion to the notebooks: identical pipeline (same
`src.lstm_utils` engine, same three-way split, same checkpoint bundles), but no
Jupyter needed. It's meant for the GPU box, where you want to launch runs or
sweeps without opening a notebook.

Examples
--------
Univariate, GPU with mixed precision::

    uv run python -m src.train --variant univariate --amp --epochs 300 \\
        --hidden 128 --batch 256 --out models/04_basic_univariate.pt

Multivariate (needs the ERA5 parquet store), 7-day horizon::

    uv run python -m src.train --variant multivariate --horizon 7 \\
        --hidden 128 --amp --out models/06_seq2seq_multi.pt

Every run calls :func:`ensure_extent_data`, so a fresh machine bootstraps its own
data before training.
"""

from __future__ import annotations

import argparse

from .data_bootstrap import ensure_extent_data
from . import lstm_utils as L

# Climate feature set used by the multivariate variant (matches notebook 05/06).
CLIMATE_FEATURES = [
    "extent_mkm2", "t2m_mean", "t2m_std", "msl_mean", "msl_std",
    "wind_speed_mean", "wind_speed_std",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an Arctic sea ice LSTM.")
    p.add_argument("--variant", choices=["univariate", "multivariate"], default="univariate",
                   help="univariate reads extent from the DB; multivariate needs ERA5 parquet.")
    p.add_argument("--out", default="models/lstm.pt", help="checkpoint bundle output path")
    # Split (calendar years; val is held out from the training era, test is later).
    p.add_argument("--train-years", nargs=2, type=int, default=[1989, 2014])
    p.add_argument("--val-years", nargs=2, type=int, default=[2015, 2019])
    p.add_argument("--test-years", nargs=2, type=int, default=[2020, 2023])
    # Model / windowing.
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cyclical", action="store_true", help="add sin/cos day-of-year features")
    # Optimization.
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-download", action="store_true", help="fail instead of downloading missing data")
    return p.parse_args(argv)


def _inclusive(pair):
    """Turn an inclusive [start, end] year pair into a range."""
    return range(pair[0], pair[1] + 1)


def main(argv=None) -> None:
    args = parse_args(argv)

    ensure_extent_data(download=not args.no_download)
    L.set_seed(args.seed)
    device = L.get_device()

    config = L.TrainConfig(
        sequence_length=args.seq_len, forecast_horizon=args.horizon,
        hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout,
        batch_size=args.batch, num_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
        amp=args.amp, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    train_years = _inclusive(args.train_years)
    val_years = _inclusive(args.val_years)
    test_years = _inclusive(args.test_years)
    all_years = range(args.train_years[0], args.test_years[1] + 1)

    if args.variant == "univariate":
        from .data_utils import load_extent_daily
        df = load_extent_daily(years=all_years)
        train_df, val_df, _ = L.temporal_split(df, train_years, val_years, test_years)
        ds_kwargs = dict(features=["extent_mkm2"], add_cyclical_time=args.cyclical)
    else:
        from .data_utils import load_data
        train_df = load_data(regions="pan_arctic", years=train_years)
        val_df = load_data(regions="pan_arctic", years=val_years)
        ds_kwargs = dict(features=CLIMATE_FEATURES, add_cyclical_time=args.cyclical)

    train_ds = L.SequenceDataset(train_df, config.sequence_length, config.forecast_horizon, **ds_kwargs)
    val_ds = L.SequenceDataset(val_df, config.sequence_length, config.forecast_horizon,
                               scaler=train_ds.scaler, **ds_kwargs)

    model = L.IceExtentLSTM(
        input_size=len(train_ds.features), hidden_size=config.hidden_size,
        num_layers=config.num_layers, forecast_horizon=config.forecast_horizon,
        dropout=config.dropout,
    )
    extra = {
        "scaler": train_ds.scaler, "features": train_ds.features, "target": "extent_mkm2",
        "input_size": len(train_ds.features),
        "split": {"train": args.train_years, "val": args.val_years, "test": args.test_years},
    }

    history = L.train_model(
        model,
        L.make_loader(train_ds, config, shuffle=True),
        L.make_loader(val_ds, config, shuffle=False),
        config, device, checkpoint_path=args.out, bundle_extra=extra,
    )
    print(f"Saved best checkpoint -> {args.out} "
          f"(val {history['best_val_loss']:.6f}, {history['epochs_trained']} epochs)")


if __name__ == "__main__":
    main()
