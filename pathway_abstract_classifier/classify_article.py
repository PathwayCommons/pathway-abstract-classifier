# Import Libraries
import ktrain
import sklearn
from cached_path import cached_path

# Load model
model_path = cached_path("https://github.com/PathwayCommons/pathway-abstract-classifier/releases/download/pretrained-models/title_abstract_model.zip", extract_archive=True)
model = ktrain.load_predictor(model_path)

# Function to classify articles in terms of whether they belong in Biofactoid

# Test
if __name__ == "__main__":
    print(model.predict("Testing, testing"))