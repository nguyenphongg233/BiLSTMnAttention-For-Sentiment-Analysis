"""Shared, stable training pipeline for 5-class Amazon review classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, regularizers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    SimpleRNN,
    SpatialDropout1D,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass(frozen=True)
class Config:
    max_words: int = 30_000
    max_len: int = 160
    embedding_dim: int = 96
    rnn_units: int = 64
    lstm_units: int = 64
    dropout: float = 0.4
    num_classes: int = 5
    batch_size: int = 64
    epochs: int = 30
    validation_split: float = 0.2
    random_state: int = 42
    l2_rate: float = 1e-4
    learning_rate: float = 3e-4


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    tokenizer: Tokenizer
    vocab_size: int
    class_weight: dict[int, float]


CONFIG = Config()
LABEL_NAMES = ["1 sao", "2 sao", "3 sao", "4 sao", "5 sao"]


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    # Giữ apostrophe để không biến "not good" và các contraction thành tín hiệu nhiễu.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rating(value: object) -> int | None:
    match = re.search(r"(?<!\d)([1-5])(?:\s*(?:/|out of)\s*5)?", str(value))
    return int(match.group(1)) if match else None


def resolve_csv_path(csv_path: str | None) -> Path:
    if csv_path:
        path = Path(csv_path)
    else:
        import kagglehub

        dataset_dir = Path(
            kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")
        )
        path = dataset_dir / "Amazon_Reviews.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    return path


def load_dataframe(csv_path: str | None = None) -> pd.DataFrame:
    path = resolve_csv_path(csv_path)
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    required = {"Review Text", "Rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset phải có các cột {sorted(required)}; nhận được {list(df)}")

    df = df[["Review Text", "Rating"]].rename(
        columns={"Review Text": "text", "Rating": "rating"}
    )
    df["text_clean"] = df["text"].map(clean_text)
    df["rating"] = df["rating"].map(parse_rating)
    df = df.dropna(subset=["rating"])
    df = df[df["text_clean"].str.len() >= 3].copy()
    df["rating"] = df["rating"].astype("int32")

    # Exact duplicates làm validation quá lạc quan; nhãn xung đột thì không đáng tin để học.
    df = df.drop_duplicates(subset=["text_clean", "rating"])
    conflicting = df.groupby("text_clean")["rating"].nunique()
    conflicting_texts = conflicting[conflicting > 1].index
    df = df[~df["text_clean"].isin(conflicting_texts)].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Dataset còn quá ít mẫu hợp lệ sau khi làm sạch.")
    class_counts = df["rating"].value_counts()
    missing = sorted(set(range(1, CONFIG.num_classes + 1)) - set(class_counts.index))
    too_small = class_counts[class_counts < 2].to_dict()
    if missing or too_small:
        raise ValueError(
            "Mỗi lớp cần ít nhất 2 mẫu để stratified split; "
            f"lớp thiếu={missing}, lớp quá ít={too_small}"
        )
    return df


def prepare_data(csv_path: str | None = None, config: Config = CONFIG) -> PreparedData:
    df = load_dataframe(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=config.validation_split,
        random_state=config.random_state,
        stratify=df["rating"],
    )

    # Chỉ học vocabulary từ train để validation thực sự là dữ liệu chưa nhìn thấy.
    tokenizer = Tokenizer(num_words=config.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text_clean"])
    vocab_size = min(len(tokenizer.word_index) + 1, config.max_words)

    def encode(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sequences = tokenizer.texts_to_sequences(frame["text_clean"])
        x = pad_sequences(
            sequences,
            maxlen=config.max_len,
            padding="post",
            truncating="post",
        )
        y = frame["rating"].to_numpy(dtype="int32") - 1
        return x, y

    x_train, y_train = encode(train_df)
    x_val, y_val = encode(val_df)
    classes = np.arange(config.num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # Chặn weight cực đoan để minority class không gây gradient spike.
    weights = np.clip(weights, 0.5, 3.0)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    print(f"Train: {len(x_train):,} | Validation: {len(x_val):,}")
    print("Train class counts:", np.bincount(y_train, minlength=config.num_classes))
    print("Class weights:", {k: round(v, 3) for k, v in class_weight.items()})
    return PreparedData(
        x_train, x_val, y_train, y_val, tokenizer, vocab_size, class_weight
    )


@tf.keras.utils.register_keras_serializable(package="sentiment")
class MaskedAttention(Layer):
    def __init__(self, attention_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.supports_masking = True
        self.projection = Dense(attention_dim, activation="tanh")
        self.score = Dense(1, use_bias=False)

    def call(self, inputs, mask=None):
        logits = tf.squeeze(self.score(self.projection(inputs)), axis=-1)
        if mask is not None:
            logits = tf.where(mask, logits, tf.cast(-1e9, logits.dtype))
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return {**super().get_config(), "attention_dim": self.attention_dim}


@tf.keras.utils.register_keras_serializable(package="sentiment")
class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_len: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.supports_masking = True
        self.token_embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.position_embedding = Embedding(max_len, embed_dim)

    def call(self, token_ids):
        length = tf.shape(token_ids)[-1]
        positions = self.position_embedding(tf.range(length))
        return self.token_embedding(token_ids) + positions

    def compute_mask(self, token_ids, mask=None):
        return tf.not_equal(token_ids, 0)

    def get_config(self):
        return {
            **super().get_config(),
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }


@tf.keras.utils.register_keras_serializable(package="sentiment")
class TransformerBlock(Layer):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.supports_masking = True
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation=tf.keras.activations.gelu),
                Dropout(dropout),
                Dense(embed_dim),
            ]
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, tf.newaxis, :]
        attended = self.attention(
            inputs,
            inputs,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(inputs + self.dropout1(attended, training=training))
        return self.norm2(x + self.dropout2(self.ffn(x, training=training), training=training))

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        return {
            **super().get_config(),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        }


def compile_model(model: Model, learning_rate: float) -> Model:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_simple_rnn(vocab_size: int, config: Config = CONFIG) -> Model:
    inputs = Input((config.max_len,), dtype="int32", name="tokens")
    x = Embedding(vocab_size, config.embedding_dim, mask_zero=True)(inputs)
    x = SpatialDropout1D(config.dropout)(x)
    x = SimpleRNN(
        config.rnn_units,
        dropout=0.2,
        kernel_regularizer=regularizers.l2(config.l2_rate),
    )(x)
    x = Dense(48, activation="relu", kernel_regularizer=regularizers.l2(config.l2_rate))(x)
    x = Dropout(config.dropout)(x)
    outputs = Dense(config.num_classes, activation="softmax")(x)
    return compile_model(Model(inputs, outputs, name="SimpleRNN"), config.learning_rate)


def build_bilstm_attention(vocab_size: int, config: Config = CONFIG) -> Model:
    inputs = Input((config.max_len,), dtype="int32", name="tokens")
    x = Embedding(vocab_size, config.embedding_dim, mask_zero=True)(inputs)
    x = SpatialDropout1D(config.dropout)(x)
    x = Bidirectional(
        LSTM(
            config.lstm_units,
            return_sequences=True,
            dropout=0.2,
            kernel_regularizer=regularizers.l2(config.l2_rate),
        )
    )(x)
    x = MaskedAttention(attention_dim=64)(x)
    x = Dense(48, activation="relu", kernel_regularizer=regularizers.l2(config.l2_rate))(x)
    x = Dropout(config.dropout)(x)
    outputs = Dense(config.num_classes, activation="softmax")(x)
    return compile_model(
        Model(inputs, outputs, name="BiLSTM_Attention"), config.learning_rate
    )


def build_transformer(vocab_size: int, config: Config = CONFIG) -> Model:
    inputs = Input((config.max_len,), dtype="int32", name="tokens")
    x = TokenAndPositionEmbedding(
        config.max_len, vocab_size, config.embedding_dim
    )(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = TransformerBlock(
        embed_dim=config.embedding_dim,
        num_heads=4,
        ff_dim=192,
        dropout=0.25,
    )(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="gelu", kernel_regularizer=regularizers.l2(config.l2_rate))(x)
    x = Dropout(config.dropout)(x)
    outputs = Dense(config.num_classes, activation="softmax")(x)
    return compile_model(
        Model(inputs, outputs, name="Transformer"), config.learning_rate * 0.67
    )


MODEL_BUILDERS = {
    "rnn": build_simple_rnn,
    "bilstm": build_bilstm_attention,
    "transformer": build_transformer,
}


def train_model(
    model: Model,
    data: PreparedData,
    output_dir: Path,
    config: Config = CONFIG,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-3,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_delta=1e-3,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(output_dir / "training_log.csv"),
    ]
    return model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        class_weight=data.class_weight,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )


def evaluate_and_save(
    model: Model,
    history,
    data: PreparedData,
    output_dir: Path,
) -> dict[str, float]:
    probabilities = model.predict(data.x_val, batch_size=CONFIG.batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(data.y_val, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(data.y_val, predictions)),
        "macro_f1": float(f1_score(data.y_val, predictions, average="macro")),
    }
    report = classification_report(
        data.y_val,
        predictions,
        labels=np.arange(CONFIG.num_classes),
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    matrix = confusion_matrix(
        data.y_val, predictions, labels=np.arange(CONFIG.num_classes)
    )
    print(json.dumps(metrics, indent=2))
    print(report)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "tokenizer.json").write_text(
        data.tokenizer.to_json(), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric, axis in (("loss", axes[0]), ("accuracy", axes[1])):
        axis.plot(history.history[metric], label=f"Train {metric}")
        axis.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        axis.set_xlabel("Epoch")
        axis.set_title(metric.capitalize())
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="True", title=model.name)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def run(model_name: str, csv_path: str | None = None) -> dict[str, float]:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Model không hợp lệ: {model_name}")
    set_reproducible_seed(CONFIG.random_state)
    data = prepare_data(csv_path)
    model = MODEL_BUILDERS[model_name](data.vocab_size)
    model.summary()
    output_dir = Path("outputs") / model_name
    history = train_model(model, data, output_dir)
    metrics = evaluate_and_save(model, history, data, output_dir)
    model.save(output_dir / "final_model.keras")
    return metrics


def main(default_model: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_BUILDERS),
        default=default_model or "bilstm",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Đường dẫn Amazon_Reviews.csv; bỏ trống để tải bằng kagglehub.",
    )
    args = parser.parse_args()
    run(args.model, args.csv)


if __name__ == "__main__":
    main()
