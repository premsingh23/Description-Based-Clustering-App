import pandas as pd
from pathlib import Path


def load_csv(file) -> pd.DataFrame:
    """Load a CSV file from a path or file-like object."""
    if isinstance(file, (str, Path)):
        return pd.read_csv(file)
    return pd.read_csv(file)


def load_example() -> pd.DataFrame:
    """Load the bundled archaeology example dataset."""
    path = Path(__file__).resolve().parents[1] / "examples" / "archaeology_samples.csv"
    return pd.read_csv(path)
