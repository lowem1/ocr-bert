from xml.dom.minidom import Document
import pandas as pd
from core.transformers import DocumentTransformer
from stages.nlp_prep import DataFrameOperators
from stages.ocr_prep import OCROperators


def document_task(filepath: str) -> pd.DataFrame:
    dt: DocumentTransformer = DocumentTransformer(filepath)
    return dt.transform(OCROperators.apply_ocr).transform(
        OCROperators.create_text_dataframe
    )

def text_task(pd.DataFrame: str) -> pd.DataFrame:
    