from __future__ import annotations
import pytesseract
import pandas as pd
from PIL import Image
from typing import OrderedDict, Tuple, List, Optional


class OCROperators:
    @staticmethod
    def apply_ocr(
        document: str,
        size_override: Tuple[float, float] = None,
        psm_config: str = "--psm 6",
        oem_config: str = "--oem 3",
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
            image=document, config=f"{psm_config} {oem_config}", lang="eng"
        )
        return string_data


class OCRTransformer:
    def __init__(
        self,
        datasource: Optional["Any"],
    ):
        self._state = datasource
        self._stage_label = None

    def transform(self, func: Optional["Callable"], *args, **kwargs) -> str:
        """Function used to facilitate chained transformations of Type Document Wrapper
        input:
            - func: callable function with args
            - container: class object state transformation is being applied to
            - args: arguments
            - kwargs: keyword arugements for configuring function with multiple input parameters
        returns: Type String comprised of OCR processed data

        usage example:
            from things import do_something
            # create function
            def example_func(container = self.container, arg) -> str:
                new = do_something(arg)
                return new

            # call function

            d: DocumentTransformer = DocumentTransformer(document,None)
            output: str = d.transform(example_func, arg)

        """
        container = self._state
        self._state = func(container, *args, **kwargs)
        return self

    def run(self, stages: OrderedDict) -> OCRTransformer:
        for label, proc in stages.items():
            func, args = proc
            if isinstance(args, list):
                self.transform(func=func, *args)
            elif isinstance(args, dict):
                kwargs = args
                self.transform(func=func, **kwargs)
        return self

    def collect(
        self,
    ) -> Any:
        """Function used to access internal state of object after transformation has been applied"""
        return self._state

    @staticmethod
    def apply_ocr(
        img: Image, psm_config: str = "--psm 6", oem_config: str = "--oem 3"
    ) -> Optional["str"]:
        """
        Image Preprocessing the extracts text from image via pytesseract API
        input:
            img: filepath or remote location of image binary
            psm_config: segmentation strategy for OCR engine to predict and extract text
            oem_config: OCR Engine Mode
        returns: string document of extracted text from OCR engine

        Example Usage:
        filename: str = "some/path/to/data.png"
        output_text: str = apply_ocr(filename,size)
        # returns output text of adjusted OCR image
        """
        string_data: str = pytesseract.image_to_string(
            image=img, config=f"{psm_config} {oem_config}", lang="eng"
        )
        return string_data

    @staticmethod
    def resize_image(
        filepath: str,
        resize_factor: Optional[Tuple["int", "int"]],
    ) -> Image:
        img: Image = Image.open(filepath)
        img = img.convert("L")
        img = (
            img.resize(
                (
                    round(img.width * resize_factor[0]),
                    round(img.height * resize_factor[1]),
                )
            )
            if resize_factor
            else img
        )
        return img

    @staticmethod
    def tokenize_data(string_data: str, token_axis: Optional["int"]) -> List["str"]:
        lines: list = string_data.split("\n")
        return lines if token_axis else [line.split(" ") for line in lines]
