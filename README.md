# pathway-abstract-classifier
A tool to classify articles with pathway content in terms of whether they are suitable for [Biofactoid](https://biofactoid.org/). 

## Quickstart 
Once you have requirements (there are only 3 - see requirements.txt) installed, you can simply run:

```
# Point this to newest release to get newest model
model_path = cached_path("https://github.com/PathwayCommons/pathway-abstract-classifier/releases/download/pretrained-models/title_abstract_model.zip", extract_archive=True)

# Note that the following follows basic Ktrain (https://github.com/amaiya/ktrain) syntax. 

# Load model
model = ktrain.load_predictor(model_path)

# Predict Example 
prediction=model.predict("Article Title".strip() + ' [SEP] ' + "Article Abstract".strip())
print(prediction)
```

## Installation
```
git clone https://github.com/PathwayCommons/pathway-abstract-classifier.git
cd pathway-abstract-classifier
pip install -r requirements.txt
```

## Citing 
[Todo?] 

