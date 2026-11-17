# Predicting bank term-deposit subscription (ES 335 project)

Interactive project website (static, GitHub Pages) built on the UCI *Bank Marketing* data.

- **Live site:** `https://<your-github-username>.github.io/<repo-name>/`  <- fill in after publishing
- **Source repo:** `https://github.com/<your-github-username>/<repo-name>`
- **Author:** [your name]   (AI assistance is disclosed at the bottom of the site)

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
1. Create a repo and push this folder.
2. Settings -> Pages -> Build and deployment -> Deploy from a branch -> `main` / `/docs` -> Save.
3. Wait ~1 minute, open the URL, and paste both links into the submission.

## Layout
- `analysis/run_analysis.py` - all modelling; writes every number and chart input the site shows
- `docs/` - the website (`index.html`, `assets/`, `data/`)
- `data/` - dataset copy (CC BY 4.0) and its documentation
- `VIVA_NOTES.md` - key choices and likely "small change" questions

## Data and credits
Moro, Rita, Cortez (2014). *Bank Marketing*. UCI Machine Learning Repository, https://doi.org/10.24432/C5K306 (CC BY 4.0).
Moro, Cortez, Rita (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems* 62:22-31.
No credentials or private data are used.

## TODO before submitting
- Replace `[your name]` in `docs/index.html` and this file; fill in the two links above.
- Re-read the "AI assistance" note on the site and edit it to describe your own checks.
- Confirm the citation/licence details on the UCI dataset page.
