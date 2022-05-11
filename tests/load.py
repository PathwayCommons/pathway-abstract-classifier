import json
from pathway_abstract_classifier.pathway_abstract_classifier import Classifier
import time

classifier = Classifier()


with open('./data/citations_1000.json', 'r') as f:
    pubmed_citations = json.load(f)

num_records = len(pubmed_citations)

print(f'Start classification for {num_records} records')
start = time.time()
predictions = classifier.predict(pubmed_citations)
end = time.time()
print(f'Elapsed time: {end-start}')
