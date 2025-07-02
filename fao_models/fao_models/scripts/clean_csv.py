import pandas as pd
import os


def load_csv(filepath: str):
    df = pd.read_csv(filepath)
    return df


def add_new_index(df: pd.DataFrame):
    df["id"] = range(len(df))


def select_columns(df: pd.DataFrame, columns: list[str]):
    return df[columns]


def save(df: pd.DataFrame, location):
    df.to_csv(location, index=False, header=False)


def main():
    files = ["testing_sample.csv", "training_sample.csv", "validation_sample.csv"]
    for f in files:
        df = load_csv(f)
        add_new_index(df)
        df = select_columns(df, ["id", "lng", "lat"])
        save(df, f"match_{f}")


if __name__ == "__main__":
    main()
