"""Entry point for the regularized SimpleRNN classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentiment_pipeline import main


if __name__ == "__main__":
    main(default_model="rnn")
