"""Tiện ích dùng chung cho ba pipeline sentiment classification trên Kaggle."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@tf.keras.utils.register_keras_serializable(package="sentiment")
def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    return tf.cast(tf.abs(y_true - tf.round(y_pred)) < 1e-5, tf.float32)


DATASET_HANDLE = "dongrelaxman/amazon-reviews-dataset"
CLASS_NAMES = ["1 sao", "2 sao", "3 sao", "4 sao", "5 sao"]


class ValidationMetrics(Callback):
    """Tính các classification metric trên validation set sau mỗi epoch."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, batch_size: int):
        super().__init__()
        self.x_val = x_val
        self.y_true = y_val.astype(int) - 1
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        raw_predictions = self.model.predict(
            self.x_val,
            batch_size=self.batch_size,
            verbose=0,
        ).reshape(-1)
        y_pred = np.clip(np.round(raw_predictions), 1, 5).astype(int) - 1
        logs.update(
            {
                "val_macro_precision": float(
                    precision_score(
                        self.y_true, y_pred, average="macro", zero_division=0
                    )
                ),
                "val_macro_recall": float(
                    recall_score(
                        self.y_true, y_pred, average="macro", zero_division=0
                    )
                ),
                "val_macro_f1": float(
                    f1_score(self.y_true, y_pred, average="macro", zero_division=0)
                ),
                "val_weighted_f1": float(
                    f1_score(
                        self.y_true, y_pred, average="weighted", zero_division=0
                    )
                ),
                "val_rating_mae": float(
                    mean_absolute_error(self.y_true + 1, y_pred + 1)
                ),
            }
        )
        print(
            " — val_macro_precision: "
            f"{logs['val_macro_precision']:.4f}"
            " — val_macro_recall: "
            f"{logs['val_macro_recall']:.4f}"
            " — val_macro_f1: "
            f"{logs['val_macro_f1']:.4f}"
            " — val_weighted_f1: "
            f"{logs['val_weighted_f1']:.4f}"
            " — val_rating_mae: "
            f"{logs['val_rating_mae']:.4f}"
        )


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def print_runtime_info() -> None:
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU") or "không có")


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rating(value: object) -> int | None:
    match = re.search(r"(?<!\d)([1-5])(?:\.0)?(?!\d)", str(value).strip())
    return int(match.group(1)) if match else None


def resolve_csv_path() -> Path:
    configured = os.getenv("AMAZON_REVIEWS_CSV")
    if configured:
        path = Path(configured)
        if not path.exists():
            raise FileNotFoundError(f"AMAZON_REVIEWS_CSV không tồn tại: {path}")
        return path

    kaggle_candidates = sorted(Path("/kaggle/input").glob("**/Amazon_Reviews.csv"))
    if kaggle_candidates:
        return kaggle_candidates[0]

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Không tìm thấy dữ liệu. Hãy Add Data trên Kaggle, đặt biến "
            "AMAZON_REVIEWS_CSV, hoặc cài kagglehub."
        ) from exc

    return Path(kagglehub.dataset_download(DATASET_HANDLE)) / "Amazon_Reviews.csv"


def load_dataframe(max_samples: int | None, seed: int) -> pd.DataFrame:
    csv_path = resolve_csv_path()
    print("Dataset:", csv_path)
    df = pd.read_csv(
        csv_path, usecols=["Review Text", "Rating"], on_bad_lines="skip", engine="python"
    )
    df = df.rename(columns={"Review Text": "text", "Rating": "rating"}).dropna()
    df["rating"] = df["rating"].map(parse_rating)
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype("int32")
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "rating"])

    if max_samples and len(df) > max_samples:
        # Lấy mẫu phân tầng gần đúng để notebook thử nghiệm nhanh nhưng vẫn đủ 5 lớp.
        sampled = []
        fractions = df["rating"].value_counts(normalize=True)
        for rating, fraction in fractions.items():
            group = df[df["rating"] == rating]
            n = max(1, min(len(group), round(max_samples * fraction)))
            sampled.append(group.sample(n=n, random_state=seed))
        df = pd.concat(sampled).sample(frac=1, random_state=seed).head(max_samples)

    print(f"Số mẫu hợp lệ: {len(df):,}")
    print(df["rating"].value_counts().sort_index().to_string())
    return df.reset_index(drop=True)


def prepare_data(
    max_words: int,
    max_len: int,
    validation_split: float,
    seed: int,
    max_samples: int | None = None,
) -> dict:
    df = load_dataframe(max_samples=max_samples, seed=seed)
    labels = df["rating"].to_numpy(dtype="float32")

    # Chia trước rồi mới fit tokenizer: validation không được làm lộ vocabulary.
    train_texts, val_texts, y_train, y_val = train_test_split(
        df["text"].to_numpy(),
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
    )
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_texts),
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(val_texts),
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    print(f"Train: {len(x_train):,} | Validation: {len(x_val):,} | Vocab: {vocab_size:,}")
    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
    }


def train_model(
    model: tf.keras.Model,
    data: dict,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    use_class_weights: bool,
) -> tuple[tf.keras.callbacks.History, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ValidationMetrics(
            x_val=data["x_val"],
            y_val=data["y_val"],
            batch_size=batch_size,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    class_weight = None
    if use_class_weights:
        classes = np.unique(data["y_train"].astype(int))
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=data["y_train"].astype(int)
        )
        class_weight = dict(zip(classes.tolist(), weights.tolist()))
        print("Class weights:", class_weight)

    started = time.perf_counter()
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )
    return history, time.perf_counter() - started


def plot_history(history: tf.keras.callbacks.History, model_name: str, output_dir: Path) -> None:
    values = history.history
    epochs = np.arange(1, len(values["loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, values["loss"], marker="o", label="Train")
    axes[0, 0].plot(epochs, values["val_loss"], marker="o", label="Validation")
    axes[0, 0].set(title="Loss (MSE)", xlabel="Epoch", ylabel="MSE")

    axes[0, 1].plot(epochs, values["accuracy"], marker="o", label="Train")
    axes[0, 1].plot(epochs, values["val_accuracy"], marker="o", label="Validation")
    axes[0, 1].set(title="Accuracy", xlabel="Epoch", ylabel="Score", ylim=(0, 1))

    for key, label in [
        ("val_macro_precision", "Macro Precision"),
        ("val_macro_recall", "Macro Recall"),
    ]:
        if key in values:
            axes[0, 2].plot(epochs, values[key], marker="o", label=label)
    axes[0, 2].set(
        title="Validation Precision & Recall",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1),
    )

    for key, label in [
        ("val_macro_f1", "Macro F1"),
        ("val_weighted_f1", "Weighted F1"),
    ]:
        if key in values:
            axes[1, 0].plot(epochs, values[key], marker="o", label=label)
    axes[1, 0].set(
        title="Validation F1-score",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1),
    )

    if "val_rating_mae" in values:
        axes[1, 1].plot(
            epochs, values["val_rating_mae"], marker="o", label="Validation MAE"
        )
    axes[1, 1].set(title="Rating MAE", xlabel="Epoch", ylabel="Số sao")

    learning_rate_key = "learning_rate" if "learning_rate" in values else "lr"
    if learning_rate_key in values:
        axes[1, 2].plot(
            epochs, values[learning_rate_key], marker="o", label="Learning rate"
        )
    axes[1, 2].set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate")

    for axis in axes.flat:
        if axis.lines:
            axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle(f"{model_name} - metrics theo từng epoch", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "epoch_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(
    model: tf.keras.Model,
    data: dict,
    history: tf.keras.callbacks.History,
    model_key: str,
    model_name: str,
    output_dir: Path,
    config: dict,
    training_seconds: float,
    batch_size: int,
) -> dict:
    started = time.perf_counter()
    predictions = model.predict(data["x_val"], batch_size=batch_size, verbose=1)
    inference_seconds = time.perf_counter() - started
    y_pred_raw = predictions.flatten()
    y_pred = np.clip(np.round(y_pred_raw), 1, 5).astype(int) - 1
    y_true = data["y_val"].astype(int) - 1

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "rating_mae": float(mean_absolute_error(y_true + 1, y_pred + 1)),
        "rating_mse": float(mean_squared_error(data["y_val"], y_pred_raw)),
    }
    summary = {
        "model_key": model_key,
        "model_name": model_name,
        "parameters": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "training_seconds": float(training_seconds),
        "inference_ms_per_sample": float(inference_seconds * 1000 / len(y_true)),
        "validation_samples": int(len(y_true)),
        "config": config,
        "metrics": metrics,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index_label="epoch")
    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(
        {
            "true_rating": y_true + 1,
            "predicted_rating": y_pred + 1,
            "predicted_value": y_pred_raw,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    plot_history(history, model_name, output_dir)
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(title=f"Confusion matrix - {model_name}", xlabel="Dự đoán", ylabel="Thực tế")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def save_artifacts(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    output_dir: Path,
    model_filename: str,
) -> None:
    model.save(output_dir / model_filename)
    with (output_dir / "tokenizer.json").open("w", encoding="utf-8") as file:
        file.write(tokenizer.to_json())


def run_pipeline(
    *,
    model_key: str,
    model_name: str,
    model_filename: str,
    build_model,
    config: dict,
) -> dict:
    output_dir = Path(config["output_dir"])
    set_global_seed(config["seed"])
    print_runtime_info()
    data = prepare_data(
        max_words=config["max_words"],
        max_len=config["max_len"],
        validation_split=config["validation_split"],
        seed=config["seed"],
        max_samples=config["max_samples"],
    )
    model = build_model(data["vocab_size"], config)
    model.summary()
    history, training_seconds = train_model(
        model=model,
        data=data,
        output_dir=output_dir,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        use_class_weights=config["use_class_weights"],
    )
    summary = evaluate_and_save(
        model=model,
        data=data,
        history=history,
        model_key=model_key,
        model_name=model_name,
        output_dir=output_dir,
        config=config,
        training_seconds=training_seconds,
        batch_size=config["batch_size"],
    )
    save_artifacts(model, data["tokenizer"], output_dir, model_filename)
    print(f"Hoàn tất. Artifact được lưu tại: {output_dir.resolve()}")
    return summary


def base_config(model_key: str) -> dict:
    max_samples_value = env_int("MAX_SAMPLES", 0)
    return {
        "max_words": env_int("MAX_WORDS", 50_000),
        "max_len": env_int("MAX_LEN", 200),
        "embedding_dim": env_int("EMBEDDING_DIM", 128),
        "dropout_rate": env_float("DROPOUT_RATE", 0.3),
        "num_classes": 5,
        "batch_size": env_int("BATCH_SIZE", 128),
        "epochs": env_int("EPOCHS", 10),
        "validation_split": env_float("VALIDATION_SPLIT", 0.2),
        "seed": env_int("RANDOM_STATE", 42),
        "max_samples": max_samples_value or None,
        "use_class_weights": os.getenv("USE_CLASS_WEIGHTS", "1") == "1",
        "output_dir": os.getenv("OUTPUT_DIR", f"outputs/{model_key}"),
    }
