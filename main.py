import pandas as pd
from ml import config as cfg
from stages.nlp_prep import DataFrameOperators as dfo
from core.transformers import DocumentTransformer
from stages.ocr_prep import OCRTransformer

filepath = "/Users/michaellowe/Documents/code/git-repos/ir-document-engine/data/line_level_test.parquet"
document = "/Users/michaellowe/Downloads/ncp-1-7.png"


source -> runner(transformer) -> writer

ocr = OCRTransformer(document)


steps = {
    "stg_1": (ocr.resize_image, {"resize_factor": (2.25, 2.25)}),
    "stg_2": (ocr.apply_ocr, {"psm_config": "--psm 4"}),
    "stg_3": (ocr.tokenize_data, {"token_axis": 0}),
}

ocr.run(steps)
print(ocr.collect())