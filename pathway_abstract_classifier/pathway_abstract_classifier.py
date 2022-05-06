from typing import Any
from pydantic import BaseModel, PrivateAttr, validator

# from typing import Final
import ktrain
from cached_path import cached_path


class Classifier(BaseModel):
    """
    Classifier to identify articles with biological pathway information.

    Attributes
    ----------
    model_url : str
        A url to a previously saved Predictor instance model

    Methods
    ----------
    request(url: str, **opts)
        Make request with appropriate body form parameters
    """

    model_url: str = (
        "https://github.com/PathwayCommons/pathway-abstract-classifier/"
        "releases/download/pretrained-models/title_abstract_model.zip"
    )
    threshold: float = 0.5
    _model: Any = PrivateAttr()
    _sep_token: str = PrivateAttr()

    @validator("threshold")
    def min_threshold_is_nonneg_lt_one(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Must be float on [0, 1]")
        return v

    def __init__(self, **data: Any) -> None:
        """Initializes Classifier instance"""
        super().__init__(**data)
        model_path = cached_path(self.model_url, extract_archive=True)
        self._model = ktrain.load_predictor(model_path)
        self._sep_token = self._model.preproc.get_tokenizer().sep_token
