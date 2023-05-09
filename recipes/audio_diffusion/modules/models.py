from samantha.core import BaseModel


class DiffusionModel(BaseModel):
    def __init__(self, stages, input_names, output_names):
        super().__init__(
            stages=stages, input_names=input_names, output_names=output_names
        )
