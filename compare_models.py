"""Tạo bảng và biểu đồ so sánh từ artifact của ba pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["rnn", "bilstm_attention", "transformer"]
MODEL_LABELS = {
    "rnn": "SimpleRNN",
    "bilstm_attention": "BiLSTM + Attention",
    "transformer": "Transformer",
}


def load_runs(outputs_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows = []
    run_dirs = {}
    for model_key in MODEL_ORDER:
        run_dir = outputs_dir / model_key
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Thiếu {metrics_path}. Hãy train đủ ba model trước khi so sánh."
            )
        with metrics_path.open(encoding="utf-8") as file:
            run = json.load(file)
        rows.append(
            {
                "model_key": model_key,
                "Model": run.get("model_name", MODEL_LABELS[model_key]),
                "Accuracy": run["metrics"]["accuracy"],
                "Macro Precision": run["metrics"]["macro_precision"],
                "Macro Recall": run["metrics"]["macro_recall"],
                "Macro F1": run["metrics"]["macro_f1"],
                "Weighted F1": run["metrics"]["weighted_f1"],
                "Rating MAE": run["metrics"]["rating_mae"],
                "Rating MSE": run["metrics"].get("rating_mse", 0.0),
                "Parameters": run["parameters"],
                "Epochs": run["epochs_trained"],
                "Training seconds": run["training_seconds"],
                "Inference ms/sample": run["inference_ms_per_sample"],
            }
        )
        run_dirs[model_key] = run_dir
    return pd.DataFrame(rows), run_dirs


def save_metric_chart(results: pd.DataFrame, output_dir: Path) -> None:
    score_columns = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    long = results.melt(
        id_vars="Model",
        value_vars=score_columns,
        var_name="Metric",
        value_name="Score",
    )
    fig, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("So sánh chất lượng phân loại")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_resource_chart(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    charts = [
        ("Parameters", "Số tham số"),
        ("Training seconds", "Thời gian train (giây)"),
        ("Inference ms/sample", "Inference (ms/mẫu)"),
    ]
    for axis, (column, title) in zip(axes, charts):
        sns.barplot(data=results, x="Model", y=column, hue="Model", legend=False, ax=axis)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "resource_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1(run_dirs: dict[str, Path], output_dir: Path) -> None:
    rows = []
    for model_key, run_dir in run_dirs.items():
        report = pd.read_csv(run_dir / "classification_report.csv", index_col=0)
        for rating in range(1, 6):
            rows.append(
                {
                    "Model": MODEL_LABELS[model_key],
                    "Rating": f"{rating} sao",
                    "F1": report.loc[f"{rating} sao", "f1-score"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_class_f1.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=frame, x="Rating", y="F1", hue="Model", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("F1-score theo từng mức rating")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(run_dirs: dict[str, Path], output_dir: Path) -> None:
    charts = [
        ("val_loss", "Validation MSE", "MSE"),
        ("val_accuracy", "Validation Accuracy", "Score"),
        ("val_macro_precision", "Validation Macro Precision", "Score"),
        ("val_macro_recall", "Validation Macro Recall", "Score"),
        ("val_macro_f1", "Validation Macro F1", "Score"),
        ("val_rating_mae", "Validation Rating MAE", "Số sao"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for model_key, run_dir in run_dirs.items():
        history = pd.read_csv(run_dir / "history.csv")
        epochs = range(1, len(history) + 1)
        label = MODEL_LABELS[model_key]
        for axis, (column, _, _) in zip(axes.flat, charts):
            if column in history:
                axis.plot(epochs, history[column], marker="o", label=label)
    for axis, (column, title, ylabel) in zip(axes.flat, charts):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        if column in {
            "val_accuracy",
            "val_macro_precision",
            "val_macro_recall",
            "val_macro_f1",
        }:
            axis.set_ylim(0, 1)
        if axis.lines:
            axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "epoch_metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown(results: pd.DataFrame, output_dir: Path) -> None:
    display = results.drop(columns=["model_key"]).copy()
    for column in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["Rating MAE"] = display["Rating MAE"].map(lambda value: f"{value:.4f}")
    display["Rating MSE"] = display["Rating MSE"].map(lambda value: f"{value:.4f}")
    display["Parameters"] = display["Parameters"].map(lambda value: f"{value:,}")
    display["Training seconds"] = display["Training seconds"].map(lambda value: f"{value:.1f}")
    display["Inference ms/sample"] = display["Inference ms/sample"].map(
        lambda value: f"{value:.4f}"
    )
    best_f1 = results.loc[results["Macro F1"].idxmax(), "Model"]
    fastest = results.loc[results["Inference ms/sample"].idxmin(), "Model"]
    smallest = results.loc[results["Parameters"].idxmin(), "Model"]
    content = (
        "# Kết quả so sánh thực nghiệm\n\n"
        + display.to_markdown(index=False)
        + "\n\n"
        + f"- Macro F1 cao nhất: **{best_f1}**.\n"
        + f"- Inference nhanh nhất trong lần chạy này: **{fastest}**.\n"
        + f"- Ít tham số nhất: **{smallest}**.\n\n"
        + "> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.\n"
    )
    (output_dir / "comparison_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--comparison-dir", type=Path, default=Path("outputs/comparison")
    )
    args = parser.parse_args()
    args.comparison_dir.mkdir(parents=True, exist_ok=True)

    results, run_dirs = load_runs(args.outputs_dir)
    results.drop(columns=["model_key"]).to_csv(
        args.comparison_dir / "model_comparison.csv", index=False
    )
    save_metric_chart(results, args.comparison_dir)
    save_resource_chart(results, args.comparison_dir)
    save_per_class_f1(run_dirs, args.comparison_dir)
    save_training_curves(run_dirs, args.comparison_dir)
    save_markdown(results, args.comparison_dir)
    print(results.drop(columns=["model_key"]).to_string(index=False))
    print(f"Đã lưu báo cáo tại {args.comparison_dir.resolve()}")


if __name__ == "__main__":
    main()
