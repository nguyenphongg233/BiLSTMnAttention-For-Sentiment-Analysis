"""Pipeline Kaggle: Amazon Reviews 1-5 sao với vanilla SimpleRNN baseline."""

from __future__ import annotations

import sys
from pathlib import Path

from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_pipeline import base_config, env_int, run_pipeline


def build_model(vocab_size: int, config: dict) -> Model:
    inputs = Input(shape=(config["max_len"],), dtype="int32", name="tokens")
    x = Embedding(
        vocab_size,
        config["embedding_dim"],
        mask_zero=True,
        weights=[config["embedding_matrix"]],
        trainable=True,
        name="token_embedding",
    )(inputs)
    x = Dropout(config["dropout_rate"], name="embedding_dropout")(x)
    x = SimpleRNN(
        config["rnn_units"],
        dropout=config["dropout_rate"],
        recurrent_dropout=0.0,
        unroll=True,
        name="simple_rnn",
    )(x)
    x = Dense(
        64, 
        activation="relu", 
        kernel_regularizer=l2(config["l2_rate"]),
        name="classifier_hidden"
    )(x)
    x = Dropout(config["dropout_rate"], name="classifier_dropout")(x)
    outputs = Dense(
        config["num_classes"], 
        activation="softmax", 
        kernel_regularizer=l2(config["l2_rate"]),
        name="rating"
    )(x)
    model = Model(inputs, outputs, name="SimpleRNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    config = base_config("rnn")
    config["rnn_units"] = env_int("RNN_UNITS", 128)
    run_pipeline(
        model_key="rnn",
        model_name="SimpleRNN",
        model_filename="rnn.keras",
        build_model=build_model,
        config=config,
    )


if __name__ == "__main__":
    main()
