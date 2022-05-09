import pytest
import json
from pathway_abstract_classifier.pathway_abstract_classifier import Classifier

#############################
#   Helpers
#############################


#############################
#   Unit tests
#############################


class TestClassifierClass:
    classifier = Classifier()

    def test_constrained_threshold(self):
        with pytest.raises(ValueError):
            Classifier(threshold=1.1)


class TestClassifierInstance:
    classifier = Classifier()

    @pytest.fixture
    def pubmed_citations(self, shared_datadir):
        file = (shared_datadir / "citations.json").open()
        citations = json.load(file)
        return citations

    def test_to_texts_handles_empty(self, shared_datadir):
        file = (shared_datadir / "docs.json").open()
        documents = json.load(file)
        texts = self.classifier._to_texts(documents)
        assert len(texts) == len(documents)
        assert texts[0] == ""
        assert texts[1] != ""

    def test_predict(self, pubmed_citations):
        predictions = self.classifier.predict(pubmed_citations)
        assert len(predictions) == len(pubmed_citations)
        input_pmids = [citation["pmid"] for citation in pubmed_citations]
        for prediction in predictions:
            assert prediction.document["pmid"] in input_pmids
            assert isinstance(prediction.probability, float)
            assert prediction.classification == 0 or prediction.classification == 1
