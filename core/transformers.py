from typing import Optional, Any, Callable, Mapping, Tuple
from glob import glob


class DocumentTransformer:
    """
    Object is state based Actor that performs various sequential chained tranformations on
    documents being preprocessed for OCR inference for pretraining the OCR BERT transformer
    where each transformation stage can be abstracted away to its own functional call for testing
    and object versioning.

    Example of class definition and usage:
        dummy_container = AnyObject()
        d: DocumentTransformer = DocumentTransformer(dummy_container)
        final_output = (
            d.transform(do_something, args)
            .transform(do_more, args)
            .transform(do_a_little_more, args)
            .transform(do_are_we_done_yet, args)
            .transform(do_yep, args)
        )

    Example usage with stage labeling:
        dummy_container = AnyObject()
        d: DocumentTransformer = DocumentTransformer(dummy_container)
        my_transformation_steps = OrderedDict({
            'stage 1': (do_something, args),
            'stage 2': (do_more, args),
            'stage 3': (do_a_little_more, args)
        })
        final_output = d.apply(my_transformation_steps)


    """

    def __init__(
        self,
        container: Optional["Any"],
    ):
        self.__state__ = container
        self.__stage_label__ = None

    def apply(self, stages: Optional[Mapping["str", Tuple["Callable", "Any"]]]):
        for label, proc in stages.items():
            self.__stage_label__ = label
            func, args = proc
            print("function: ", func)
            print("args: ", args)
            self.transform(
                func=func,
            )
        return self

    def transform(
        self,
        func: Optional["Callable"],
    ) -> str:
        """Function used to facilitate chained transformations of Type Document Wrapper
        input:
            - func: callable function with args
            - container: class object state transformation is being applied to
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
        container = self.__state__
        self.__state__ = func(
            container,
        )
        return self

    def collect(
        self,
    ) -> Any:
        """Function used to access internal state of object after transformation has been applied"""
        return self.__state__
