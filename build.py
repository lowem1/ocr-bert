from asyncio import futures
import glob
import uuid
import json
import os
import sys
import pandas as pd
import ray
from ml import config as cfg
from stages.nlp_prep import DataFrameOperators as dfo
from core.transformers import OCRTransformer
from typing import List, Dict, Tuple, Any, Optional


def generate_truths(
    document_repo: Optional["str"],
    output_dir: Optional["str"],
) -> pd.DataFrame:
    dfs: List = list()
    document_repo: str = f"{document_repo}/.*txt"
    files = glob.glob(document_repo)
    get_lines: List  = lambda _: open(_).read(), os.path.basename(_).split(".")[0]
    for _file in files:
        lines =

def with_token_substitution(
    document_repo: Optional["str"] = None,
    output_dir: Optional["str"] = "/tmp/augs",
    files: Optional[List["str"]] = None,
) -> pd.DataFrame:
    future: List = list()
    if files and document_repo == None:
        documents = files
    elif files == None and document_repo:
        documents: List["str"] = glob.glob(document_repo)

    @ray.remote
    def token_substitution(filepath: str, output_filepath: str) -> List:
        df = pd.read_parquet(filepath, engine="fastparquet")
        output_fname = output_filepath
        print("Generating token file: ", output_fname)
        final_df = (
            df.pipe(dfo.with_clean_lines, "output")
            .pipe(dfo.with_mask_insertion, "output")
            .pipe(dfo.with_mask_augmenetation, "masks", "output")
        )
        final_df.to_parquet(engine="fastparquet", path=output_fname, compression="gzip")

    for document in documents:
        output_file = f"{output_dir}/{uuid.uuid1()}.parquet.gz"
        future.append(token_substitution.remote(document, output_file))
    results = ray.get(future)
    return output_dir if results else None


def with_text_extraction(
    document_repo: str, output_dir: str, runtime_conf: List[Dict["str", "Any"]]
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
    document_repo: str = f"{document_repo}/*.png"
    documents: List = glob.glob(document_repo)

    @ray.remote
    def text_extraction(
        document: str,
        resize_factor: Tuple["float", "float"] = (1, 1),
        psm_config: str = "--psm 6",
        token_axis: int = 1,
        document_tag: int = None,
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
        output_fname = f"{staging_path}/{uuid.uuid1()}.parquet.gz"
        output_record = None
        output_record = {
            "output": ocr_output,
            "document_tag": document_tag,
            "psm_config": psm_config,
            "resize_factor": str(resize_factor),
        }
        # print(output_record)

        print("Generating augmentation: ", output_fname)
        pd.DataFrame.from_dict(output_record).to_parquet(
            engine="fastparquet", path=output_fname, compression="gzip"
        )

        return output_fname

    for i, document in enumerate(documents):
        document_tag: str = os.path.basename(document).split(".")[0]
        for conf in runtime_conf:
            conf["document"] = document
            conf["document_tag"] = document_tag
            future.append(text_extraction.remote(**conf))
    results = ray.get(future)
    return results


def run_process(
    source_path: Optional["str"],
    staging_path: Optional["str"],
    sink_path: Optional["str"],
    config_map: Dict["str", "Any"],
) -> str:
    stg_1 = with_text_extraction(
        document_repo=source_path, output_dir=staging_path, runtime_conf=config_map
    )
    stg_2 = with_token_substitution(files=stg_1, output_dir=sink_path)

    return stg_2


if __name__ == "__main__":
    ray.init()

    source_path: str = sys.argv[1]
    staging_path: str = sys.argv[2]
    sink_path: str = sys.argv[3]

    print("source path: ", source_path)
    print("staging path: ", staging_path)
    print("sink path: ", sink_path)

    resize = [(0.5, 0.5), (1, 1), (2.25, 2.25), (3.25, 3.25)]
    psm = ["--psm 4", "--psm 6"]
    config_map = list()

    for _resize in resize:
        for _psm in psm:
            config_map.append({"resize_factor": _resize, "psm_config": _psm})

run_process(source_path, staging_path, sink_path, config_map)
