# Greater Bay Area Travel Choice Recommender

An academic project exploring stated-preference transport choices in the
Greater Bay Area. The core work is survey-data preparation, choice modelling,
and Value of Travel Time Savings (VTTS) analysis. A FastAPI interface was added
after the modelling work as a prototype for trying model outputs.

## Data and privacy

The survey workbook is restricted by the project supervisor and is not included
in this repository. Do not commit the workbook, respondent-level data, or other
restricted materials. The API and analysis require a locally available,
professor-approved workbook in the expected format. The mode-attribute CSV files
in `datamode/` describe the choice scenarios and are loaded by the model code.

The restricted workbook used for this project contains 1,000 respondent records
(250 in each of four questionnaire sets). Because the survey data are
confidential, the underlying records, fitted coefficients, and model performance
values cannot be shared publicly.

Set `GBA_SURVEY_PATH` to the local workbook path before starting the API:

```powershell
$env:GBA_SURVEY_PATH = "C:\path\to\approved-survey.xlsx"
```

On macOS/Linux:

```bash
export GBA_SURVEY_PATH="/path/to/approved-survey.xlsx"
```

The workbook must contain the `Full` sheet and the survey columns expected by
`model.py`. If data are unavailable, the API starts without loading it and
returns a clear service error when a data-dependent endpoint is requested.

### Run without the private workbook

For a local API demonstration, opt into deterministic synthetic data before
starting the server:

```powershell
# Windows PowerShell
$env:GBA_DEMO_MODE = "1"
uvicorn api:app --reload
```

```bash
# macOS/Linux
GBA_DEMO_MODE=1 uvicorn api:app --reload
```

Demo responses identify their source as `synthetic_demo` and include an explicit
notice. The generated records are only for exercising the API; they are not
derived from, calibrated to, or evidence for the restricted survey results.
Unset `GBA_DEMO_MODE` to return to private-survey mode.

## Model and interpretation

The analysis expands each reported alternative into chosen/not-chosen rows,
standardizes the predictors, and fits L1-regularized logistic regression models
with the `saga` solver (`C=5`, `max_iter=5000`). L1 regularization was used to
shrink less informative coefficients toward zero and encourage a sparser model.
VTTS is derived from the ratio of estimated time and fare coefficients.

The current modelling code fits each scenario model on all available records; it
does not implement a train/test split or cross-validation, and it does not
calculate out-of-sample accuracy, log-loss, or other validation metrics. These
should be added before making claims about predictive performance. Results
depend on the private survey, scenario attributes, model specification, and
sample size; they should be interpreted as academic estimates, not validated
travel advice.

`POST /predict_choice` returns a binary model probability for one requested
mode. Scores from separate mode requests are **not normalized** and may not sum
to 1. This API is a prototype; it does not provide a calibrated, joint choice
distribution over all modes.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Anon-tokyo123/Greater-Bay-Area-Travel-Choice-Recommender.git
cd Greater-Bay-Area-Travel-Choice-Recommender
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install packages and launch the local server:

```bash
python -m pip install -r requirements.txt
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.
The survey data are loaded only when `/vtts` or `/predict_choice` is called.

### Run the automated tests

The pytest suite uses synthetic demo data and does not need the restricted
workbook:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs the same suite on pushes to `main` or `recommender-*`
branches and on pull requests targeting `main`.

## Endpoints

- `GET /` — API status message.
- `GET /vtts?purpose=Work` — VTTS estimates by distance and time component.
- `POST /predict_choice?purpose=Work&distance=Short%20(50km)&mode=HSR` — binary
  chosen-vs-not-chosen score for the supplied alternative attributes.

All numeric travel attributes must be non-negative. Crowding is a proportion
from 0 to 1.

## Project scope and limitations

- The survey workbook is not published for privacy and project restrictions.
- The API was developed after the final-year modelling work and is a prototype.
- Predictions depend on the local restricted data and are not independently
  reproducible from this repository alone.
- Long-distance work-trip coverage is limited, as noted in the project analysis.
- The binary mode scores are not probabilities over a jointly evaluated set of
  alternatives.
- Future work: add a multinomial choice model or normalize scores across all
  modes with softmax so that a scenario returns one joint distribution whose
  probabilities sum to 1.
- Future work: evaluate generalization with a respondent-level holdout split or
  cross-validation, report approved aggregate metrics, and expand API test
  coverage as the model evolves. Keep all evaluation data and outputs within the
  restrictions set by the project supervisor.

## License

See [LICENSE](LICENSE).
