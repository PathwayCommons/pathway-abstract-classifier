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

    def test_predict(self, pubmed_citations):
        predictions = self.classifier.predict(pubmed_citations)
        assert len(predictions) == len(pubmed_citations)
        pmids = [citation["pmid"] for citation in pubmed_citations]
        for prediction in predictions:
            assert "pmid" in prediction
            assert prediction["pmid"] in pmids
            assert "probability" in prediction
            assert "prediction" in prediction
