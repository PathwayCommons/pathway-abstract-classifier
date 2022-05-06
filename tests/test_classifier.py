import pytest
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

    def test_instance_has_private_instances(self):
        assert self.classifier._model is not None
        assert self.classifier._sep_token is not None
