"""Demo dự đoán sentiment bằng ba model đã train."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from sentiment_pipeline import clean_text


ROOT = Path(__file__).resolve().parent
MODEL_CONFIGS = {
    "rnn": {
        "label": "SimpleRNN",
        "script": ROOT / "RNN/rnn_sentiment.py",
        "model": "rnn.keras",
    },
    "bilstm_attention": {
        "label": "BiLSTM + Attention",
        "script": ROOT / "BiLSTM +  Attention/BiLSTM_Attention.py",
        "model": "bilstm_attention.keras",
    },
    "transformer": {
        "label": "Transformer",
        "script": ROOT / "Transformer/transformer_sentiment.py",
        "model": "transformer.keras",
    },
}

DEFAULT_SAMPLES = [
    ("Terrible product. It broke after one day and support never replied.", 1),
    ("The item works, but the quality is disappointing for the price.", 2),
    ("It is okay. Nothing special, but it does what it says.", 3),
    ("Very good product, easy to use and arrived on time.", 4),
    ("Absolutely fantastic! Excellent quality and I highly recommend it.", 5),
]


def import_model_definitions() -> None:
    """Đăng ký custom layer để Keras có thể load các model đã lưu."""
    for model_key, config in MODEL_CONFIGS.items():
        spec = importlib.util.spec_from_file_location(
            f"sentiment_{model_key}", config["script"]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Không thể import {config['script']}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def load_tokenizer(path: Path):
    with path.open(encoding="utf-8") as file:
        return tokenizer_from_json(file.read())


def print_saved_metrics(model_dir: Path, label: str) -> None:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"\n{label}: chưa có metrics.json")
        return
    with metrics_path.open(encoding="utf-8") as file:
        run = json.load(file)
    metrics = run["metrics"]
    print(f"\n{label} — validation metrics")
    print(
        pd.Series(
            {
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Rating MAE": metrics["rating_mae"],
                "Rating MSE": metrics.get("rating_mse", float("nan")),
                "Inference ms/sample": run["inference_ms_per_sample"],
            }
        ).to_string(float_format=lambda value: f"{value:.4f}")
    )


def predict_samples(
    outputs_dir: Path,
    samples: list[tuple[str, int | None]],
) -> pd.DataFrame:
    rows = [
        {"sample": index, "review": text, "expected_rating": expected}
        for index, (text, expected) in enumerate(samples, start=1)
    ]
    cleaned_texts = [clean_text(text) for text, _ in samples]

    for model_key, config in MODEL_CONFIGS.items():
        model_dir = outputs_dir / model_key
        model_path = model_dir / config["model"]
        tokenizer_path = model_dir / "tokenizer.json"
        metrics_path = model_dir / "metrics.json"
        missing = [
            str(path)
            for path in (model_path, tokenizer_path, metrics_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Thiếu artifact của {config['label']}: {', '.join(missing)}"
            )

        with metrics_path.open(encoding="utf-8") as file:
            run = json.load(file)
        max_len = int(run["config"]["max_len"])
        tokenizer = load_tokenizer(tokenizer_path)
        sequences = pad_sequences(
            tokenizer.texts_to_sequences(cleaned_texts),
            maxlen=max_len,
            padding="post",
            truncating="post",
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        raw = model.predict(sequences, verbose=0).reshape(-1)
        ratings = np.clip(np.round(raw), 1, 5).astype(int)

        for row, raw_value, rating in zip(rows, raw, ratings):
            row[f"{config['label']} value"] = float(raw_value)
            row[f"{config['label']} rating"] = int(rating)
            if row["expected_rating"] is not None:
                row[f"{config['label']} abs_error"] = abs(
                    int(row["expected_rating"]) - int(rating)
                )
        print_saved_metrics(model_dir, config["label"])

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo một vài review mẫu bằng cả ba sentiment model."
    )
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--text",
        action="append",
        help="Review tùy chỉnh; có thể truyền nhiều lần. Khi dùng, expected rating để trống.",
    )
    args = parser.parse_args()

    import_model_definitions()
    samples = (
        [(text, None) for text in args.text]
        if args.text
        else DEFAULT_SAMPLES
    )
    results = predict_samples(args.outputs_dir, samples)

    print("\nDự đoán trên các sample")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_colwidth",
        70,
        "display.width",
        240,
    ):
        print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
