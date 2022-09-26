from transformers import AutoTokenizer, AutoModel, AdamW, pipeline
from typing import Optional, Tuple, List


def load_checkpoint(
    arg: Optional["str"],
) -> Optional[Tuple[AutoModel, AutoTokenizer]]:
    return (
        AutoModel.from_pretrained(arg),
        AutoTokenizer.from_pretrained(arg, padding=True),
    )


DEFAULT_CHECKPOINT = "bert-base-uncased"
CLINCIAL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
BIO_BERT_CHECKPOINT = "dmis-lab/biobert-v1.1"
# BERT_BASE, BERT_BASE_TOKENIZER = load_checkpoint(
#     DEFAULT_CHECKPOINT,
# )
# BIO_BERT_BASE, BIO_BERT_BASE_TOKENIZER = load_checkpoint(
#     BIO_BERT_CHECKPOINT,
# )
# BIO_CLINCIAL_BERT_BASE, BIO_CLINICAL_BERT_BASE_TOKENIZER = load_checkpoint(
#     CLINCIAL_CHECKPOINT,
# )

# MASKING_PIPELINE = pipeline(
#     "text-generation", model=BIO_CLINCIAL_BERT_BASE, tokenizer=BIO_BERT_BASE_TOKENIZER
# )
