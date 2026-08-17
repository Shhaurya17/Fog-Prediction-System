"""
Main Command Line Entry Point for Fog Prediction System.
Orchestrates raw data compilation, missing value imputation, feature engineering, sequential tensor generation, model training, and metric evaluation.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .config import (
    COMPILED_DATASET_PATH,
    D_FF,
    EDA_OUTPUT_DIR,
    EPOCHS,
    H_DIM,
    IMPUTED_DATASET_PATH,
    LEARNING_RATE,
    MODEL_OUTPUT_DIR,
    MODEL_READY_DATASET_PATH,
    N_HEADS,
    SEED,
    SEQ_LEN,
    TOP_K_FEATURES,
)
from .data import DatasetBuilder, load_dataset
from .models import NumPyLSTMTransformer
from .processing import (
    FeatureEngineer,
    create_sequences,
    process_missing_values,
    temporal_train_val_test_split,
)
from .training import ModelEvaluator, NumPyTrainer
from .training.trainer import predict_batched
from .visualization import (
    plot_cyclic_encodings,
    plot_learning_curves,
    plot_missing_values,
    plot_top_features_correlation,
)


def run_full_pipeline():
    """Runs the complete end-to-end machine learning pipeline."""
    np.random.seed(SEED)
    print("=" * 60)
    print(" 🌫️ STARTING FOG PREDICTION PIPELINE (LSTM + TRANSFORMER) ")
    print("=" * 60)

    # 1. Dataset Loading or Creation
    if not COMPILED_DATASET_PATH.exists():
        print("Compiled dataset not found. Building dataset from raw sources...")
        builder = DatasetBuilder()
        raw_df = builder.build()
    else:
        raw_df = load_dataset(COMPILED_DATASET_PATH)

    plot_missing_values(raw_df, save_path=EDA_OUTPUT_DIR / "missing_values.png")

    # 2. EDA & Imputation
    df_imputed = process_missing_values(raw_df, output_path=IMPUTED_DATASET_PATH)

    # 3. Feature Engineering
    engineer = FeatureEngineer(top_k=TOP_K_FEATURES)
    df_ready, selected_features = engineer.process(
        df_imputed, output_path=MODEL_READY_DATASET_PATH
    )

    plot_cyclic_encodings(save_path=EDA_OUTPUT_DIR / "cyclic_encodings.png")
    plot_top_features_correlation(
        df_ready, selected_features, save_path=EDA_OUTPUT_DIR / "top_features_corr.png"
    )

    # 4. Temporal Split & Scaling
    split_data = temporal_train_val_test_split(df_ready, selected_features)
    X_tr = split_data["X_train"]
    X_vl = split_data["X_val"]
    X_ts = split_data["X_test"]

    # 5. Build 3D Sequences
    Xtr6, ytr6 = create_sequences(X_tr, split_data["y6_train"], SEQ_LEN)
    Xvl6, yvl6 = create_sequences(X_vl, split_data["y6_val"], SEQ_LEN)
    Xts6, yts6 = create_sequences(X_ts, split_data["y6_test"], SEQ_LEN)

    Xtr12, ytr12 = create_sequences(X_tr, split_data["y12_train"], SEQ_LEN)
    Xvl12, yvl12 = create_sequences(X_vl, split_data["y12_val"], SEQ_LEN)
    Xts12, yts12 = create_sequences(X_ts, split_data["y12_test"], SEQ_LEN)

    in_dim = len(selected_features)

    # 6. Train 6-Hour Model
    model6 = NumPyLSTMTransformer(in_dim=in_dim, h_dim=H_DIM, n_heads=N_HEADS, d_ff=D_FF)
    trainer6 = NumPyTrainer(model6, lr=LEARNING_RATE, epochs=EPOCHS)
    hist6 = trainer6.train(Xtr6, ytr6, Xvl6, yvl6, name="6-Hour")

    preds6_test = predict_batched(model6, Xts6)
    metrics6 = ModelEvaluator.evaluate(yts6, preds6_test)
    ModelEvaluator.print_summary(metrics6, name="6-Hour Model (Test Set)")

    # 7. Train 12-Hour Model
    model12 = NumPyLSTMTransformer(in_dim=in_dim, h_dim=H_DIM, n_heads=N_HEADS, d_ff=D_FF)
    trainer12 = NumPyTrainer(model12, lr=LEARNING_RATE, epochs=EPOCHS)
    hist12 = trainer12.train(Xtr12, ytr12, Xvl12, yvl12, name="12-Hour")

    preds12_test = predict_batched(model12, Xts12)
    metrics12 = ModelEvaluator.evaluate(yts12, preds12_test)
    ModelEvaluator.print_summary(metrics12, name="12-Hour Model (Test Set)")

    # 8. Learning Curves
    plot_learning_curves(hist6, hist12, save_path=MODEL_OUTPUT_DIR / "learning_curves.png")

    print("\n✅ PIPELINE EXECUTION COMPLETE! All results & plots saved to outputs/")


def main():
    parser = argparse.ArgumentParser(description="Fog Prediction System CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "build_data", "eda", "train"],
        help="Pipeline execution mode.",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "build_data":
        builder = DatasetBuilder()
        builder.build()
    elif args.mode == "eda":
        df = load_dataset(COMPILED_DATASET_PATH)
        process_missing_values(df)
    elif args.mode == "train":
        run_full_pipeline()


if __name__ == "__main__":
    main()
