import pandas as pd
import glob
import sys
import pytesseract
from PIL import Image


if __name__ == "__main__":
    # filepath = f"{sys.argv[1]}/*.parquet.gz"
    # print(filepath)
    # training_df: pd.DataFrame = pd.concat(
    #     list(map(lambda _: pd.read_parquet(_), glob.glob(filepath)))
    # ).query("psm_config != '--psm 4'")

    # training_df.sort_values("document_tag")
    # print(training_df.head(50))

    files = glob.glob(f"{sys.argv[1]}/*.png")
    for _file in files:
        img = Image.open(_file)
        string = pytesseract.image_to_string(
            img,
            config="--psm 6",
        )
        print(string)
