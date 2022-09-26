import pandas as pd
from typing import Optional, Tuple, List
from transformers import pipeline, AutoModel, AutoTokenizer


def with_sequence_strings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    pass


def with_token_strings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    pass


def with_clean_lines(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(
        lambda _: None if len(_) == 0 or str.isalnum(_) else _
    )
    df.dropna(inplace=True)
    return df


def with_mask_insertion(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def _(sequence: Optional["str"]) -> List:
        perms: List = list()
        tokens: List = sequence.split(" ")
        for i, data in enumerate(tokens):
            _: list = sequence.split(" ")
            _[i] = "[MASK]"
            perms.append(" ".join(_))
        return perms

    permutations: List = list(map(_, df[column]))
    df["masks"] = permutations
    return df


def with_mask_augmenetation(df: pd.DataFrame, column: str) -> pd.DataFrame:
    MASKING_PIPELINE = pipeline(
        "fill-mask",
        model="bert-base-uncased",
    )

    def _(sequence: Optional["str"]) -> str:

        aug: list = [x["sequence"] for x in MASKING_PIPELINE(sequence)]
        return aug

    df["generative_augs"] = list(map(_, df[column]))
    df = df.explode("generative_augs")
    return df