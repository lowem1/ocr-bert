import pandas as pd
from ml import config as cfg
from stages.nlp_prep import (
    with_mask_insertion,
    with_mask_augmenetation,
    with_clean_lines,
)


filepath = "/Users/michaellowe/Documents/code/git-repos/ir-document-engine/data/line_level_test.parquet"

df = pd.read_parquet(filepath)
df = df[df.document_num == 3]
print("pre-tranformed count: ", df.count())

test = (
    df.pipe(with_clean_lines, "raw_text")
    .pipe(with_mask_insertion, "raw_text")
    .explode("masks")
    .pipe(with_mask_augmenetation, "masks")
)

print("post-tranformed count: ", test.count())

print(test)
