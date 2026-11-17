# Predicting bank term-deposit subscription (ES 335 project)

Interactive project website (static, GitHub Pages) built on the UCI *Bank Marketing* data.

- **GitHub Pages site:** `https://bhukyabhavyasree.github.io/bank-deposit-prediction-project/`
- **Source repo:** `https://github.com/bhukyabhavyasree/bank-deposit-prediction-project`
- **Author:** Bhukya Bhavyasree

## Question
Using only information available before a call, can we rank clients by likelihood of subscribing to a term
deposit, and does the ranking hold on a later period? Primary split is chronological 60/20/20.
Headline: logistic regression (test ROC-AUC 0.695, PR-AUC 0.491) roughly ties a one-line "previous campaign
succeeded" rule on PR-AUC (0.480); gradient boosting is lower (0.452). Strong drift in the subscription rate
(4.8% train, 30.8% test) dominates; a random split would have reported AUC ~0.80.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analysis/run_analysis.py data/bank-additional-full.csv docs/data/results.json   # ~4 min, seed 42
python -m http.server -d docs 8000                                                      # open http://localhost:8000
```
`docs/data/results.js` is the same payload as `results.json` so the page also opens from a double-click.

## Publish on GitHub Pages
1. In Settings -> Pages -> Build and deployment, choose Deploy from a branch, then `main` / `/docs`, and save.
2. Wait for the site to publish, then open the Pages URL above and verify it loads.

## Layout
- `analysis/run_analysis.py` - all modelling; writes every number and chart input the site shows
- `docs/` - the website (`index.html`, `assets/`, `data/`)
- `data/` - dataset copy (CC BY 4.0) and its documentation

## Data and credits
Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. Licensed under CC BY 4.0.

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22-31.
No credentials or private data are used.

## AI use
Claude (Anthropic) helped draft the analysis script and website code and text, and suggested checks such as the leakage review of `duration` and `campaign`. GitHub Copilot assisted with reviewing the assignment requirements, repository setup, and project documentation. The site's reported results come from the analysis script and its output file.
