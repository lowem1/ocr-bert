from asyncio import futures
import glob
import uuid
import json
import os
import pandas as pd
import ray
from ml import config as cfg
from stages.nlp_prep import DataFrameOperators as dfo
from core.transformers import OCRTransformer
from typing import List, Dict, Tuple, Any, Optional

ray.init()


def with_token_substitution(document_repo: Optional["str"]) -> pd.DataFrame:
    future: List = list()
    documents: List["str"] = glob.glob(document_repo)

    @ray.remote
    def token_substitution(filepath: str) -> List:
        df = pd.read_parquet(filepath, engine="fastparquet")
        return (
            df.pipe(dfo.with_clean_lines, "output")
            .pipe(dfo.with_mask_insertion, "output")
            .pipe(dfo.with_mask_augmenetation, "masks", "output")
        )

    for document in documents:
        future.append(token_substitution.remote(document))
    results = ray.get(future)
    return results


def with_text_extraction(
    document_repo: str, runtime_conf: List[Dict["str", "Any"]]
) -> List:
    """Remote Function used to faciliate document text extraction and augmentation pipeline
    input:
        - document: location of document to have augmentation applied
        - resize_fact: in the preprocessing step of image processing adjust resolution
        - psm_config: level of segmentation used to extract text
        - token_axis: lines vs single term tokenization
    output:
        - list of augmented tokens
    """
    future: List = list()
    documents: List = glob.glob(document_repo)

    @ray.remote
    def text_extraction(
        document: str,
        resize_factor: Tuple["float", "float"] = (1, 1),
        psm_config: str = "--psm 6",
        token_axis: int = 1,
    ) -> List:
        ocr: OCRTransformer = OCRTransformer(document)
        steps: Dict = {
            "stg_resize": (
                ocr.resize_image,
                {"resize_factor": resize_factor},
            ),
            "stg_ocr": (
                ocr.apply_ocr,
                {"psm_config": psm_config},
            ),
            "stg_parse": (
                ocr.tokenize_data,
                {"token_axis": token_axis},
            ),
        }
        ocr.run(steps)
        ocr_output = ocr.collect()
        dirname = f"{os.path.dirname(document)}/augs"
        basename = os.path.basename(document).split(".")[0]
        output_fname = f"{dirname}/{basename}-{uuid.uuid1()}.parquet.gz"
        output_record = None
        output_record = {
            "output": ocr_output,
        }
        # print(output_record)

        print("Generating augmentation: ", output_fname)
        pd.DataFrame.from_dict(output_record).to_parquet(
            engine="fastparquet", path=output_fname, compression="gzip"
        )
        # with open(output_fname, "w") as f:
        #     f.write(json.dumps(output_record).encode("utf-8"))

        return output_fname

    for document in documents:
        for conf in runtime_conf:
            conf["document"] = document
            future.append(text_extraction.remote(**conf))
    results = ray.get(future)
    return results


# resize = [(0.5, 0.5), (1, 1), (2.25, 2.25), (3.25, 3.25)]
# psm = ["--psm 4", "--psm 6"]
# config_map = list()

# for _resize in resize:
#     for _psm in psm:
#         config_map.append({"resize_factor": _resize, "psm_config": _psm})


# repo = "/Users/michaellowe/Documents/datasets/ncps/*.png"
# test_repo = "/tmp/test/ncp/*.png"


# with_text_extraction(repo, config_map)

output = with_token_substitution("/Users/michaellowe/Documents/datasets/ncps/augs/*.gz")
