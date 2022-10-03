from __future__ import annotations
from typing import (
    Optional,
    Any,
    Callable,
    Mapping,
    Tuple,
    Dict,
)
from glob import glob
from unittest.mock import Base
from PIL import Image
import pytesseract


class BaseTransformer:
    def __init__(
        self,
        container: Optional["Any"],
    ):
        self._state = container
        self._stage_label = None

    def transform(self, func: Optional["Callable"], *args, **kwargs) -> BaseTransformer:
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

    def run(
        self, stages: Dict["str", Tuple("Callable", Dict["str", "str"])]
    ) -> BaseTransformer:
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


class OCRTransformer(BaseTransformer):
    def __init__(
        self,
        datasource: Optional["Any"],
    ):
        super().__init__(datasource)

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
