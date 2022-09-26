import pytesseract
from io import BytesIO
from PIL import Image
from typing import Tuple, List


def apply_ocr(
    document: str,
    size_override: Tuple[float, float] = None,
    psm_config: str = "--psm 6",
    oem_config: str = "--oem 2",
) -> str:
    """
    Image Preprocessing Pipeline that applied resizing, normalization, and grayscale to each image
    imput:
        document: filepath or remote location of image binary
        size_override: resize factor or scale for image to be adjusted to for processing
        psm_config: segmentation strategy for OCR engine to predict and extract text
        oem_config: OCR Engine Mode
    returns: string document of extracted text from OCR engine

    Example Usage:
    filename: str = "some/path/to/data.png"
    size: Tuple[float, float] = (1.25, 1.25)
    output_text: str = apply_ocr(filename,size)
    # returns output text of adjusted OCR image
    """
    img: Image = Image.open(document)
    img = img.convert("L")
    img = (
        img.resize(
            round(img.width * size_override[0]),
            round(img.height * size_override[1]),
        )
        if size_override
        else img
    )
    string_data: str = pytesseract.image_to_string(
        filepath=document, config=f"{psm_config} {oem_config}"
    )
    return string_data
