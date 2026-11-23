import os
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DEFAULT_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")


def download(path=DEFAULT_PATH, series="dataset", start="2000-01-01"):
    df = pd.read_csv(FRED_URL.format(series=series), index_col=0, parse_dates=True, na_values=".")
    df.index.name = "date"
    df.columns = [series]
    df = df.loc[start:].dropna()
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(os.path.join(BASE_DIR, "data", "dataset.csv"))
    print(f"Salvate {len(df)} osservazioni in: {path}")
    return path


def load(path=DEFAULT_PATH):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(float)
    s.index.freq = pd.infer_freq(s.index)
    return s



if __name__ == "__main__":
    download()