from typing import Any, List, Dict
from pydantic import BaseModel, PrivateAttr, validator
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
    predict(self, documents: List[Dict[str, str]]) -> List[Dict[str,str]]
        Make predictions based upon the text information in incoming docuemntss
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

    def _to_texts(self, documents: List[Dict[str, str]]) -> List[str]:
        """Map the title and text fields to a single string"""
        texts = [" ".join([doc["title"], self._sep_token, doc["abstract"]]) for doc in documents]
        return texts

    def _to_predictions(
        self, documents: List[Dict[str, str]], probabilities: List[float]
    ) -> List[Dict[str, Any]]:
        """Format the incoming data with the prediction results"""
        results = []
        for index, document in enumerate(documents):
            prediction = {
                "pmid": document["pmid"],
                "class": int(probabilities[index] >= self.threshold),
                "probability": float(probabilities[index]),
            }
            results.append(prediction)
        return results

    def predict(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Predictions based on text in documents"""
        results = []
        texts = self._to_texts(documents)
        probabilities = self._model.predict_proba(texts)[:,1]
        results = self._to_predictions(documents, probabilities)
        return results
