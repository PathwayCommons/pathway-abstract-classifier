#! /bin/bash
echo "Tests"
poetry run pytest tests --cov ./pathway_abstract_classifier

echo "Linting"
poetry run flake8 ./pathway_abstract_classifier --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run flake8 ./pathway_abstract_classifier --count --exit-zero --max-complexity=10 --statistics

# echo "Type checking"
# poetry run mypy . --cache-dir=/dev/null