import pandas as pd
import glob
import sys


if __name__ == "__main__":
    filepath = f"{sys.argv[1]}/*.parquet.gz"
    print(filepath)
    training_df: pd.DataFrame = pd.concat(
        list(map(lambda _: pd.read_parquet(_), glob.glob(filepath)))
    )

    print(training_df.head())
