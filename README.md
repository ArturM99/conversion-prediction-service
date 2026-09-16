# Conversion Prediction Model

The model predicts the probability of a target action (conversion) by a website user, based on Google Analytics session data (UTM tags, device, geo, visit time).

The target action is the occurrence of one of the following events within a session:
```text
sub_car_claim_click
sub_car_claim_submit_click
sub_open_dialog_click
sub_custom_question_submit_click
sub_call_number_click
sub_callback_submit_click
sub_submit_success
sub_car_request_submit_click
```
## Project structure
```text
pipeline.py       # data loading, feature engineering, sklearn pipeline assembly
train.py          # model training, saves ga_model.pkl
main.py           # FastAPI inference service
ga_model.pkl      # trained model (created by train.py)
ga_sessions.*     # source session data (not included in the repo)
ga_hits*.*        # source event data (not included in the repo)
requirements.txt
```
## Installation
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Minimal dependencies:

```
fastapi
uvicorn
pandas
numpy
scikit-learn
dill
pyarrow
```
## Data

Before training, place the session and event files in the project root.
The data loader (`read_data_file` in `pipeline.py`) finds them automatically by filename prefix, regardless of:
- file format — `.parquet`, `.pkl`, `.csv` are supported (checked in this exact order — if several formats of the same file are present, the first one found is used);
- suffix/number in the filename — `ga_hits-001.parquet`, `ga_hits-002.parquet`, `ga_hits_final.csv`, etc. all work.

Expected prefixes:

- `ga_sessions...` — session data (UTM tags, device, geo, visit date/time, etc.)
- `ga_hits...` — event data (`session_id`, `event_action`), used to build the target

If a file isn't found in any of the supported formats, `train.py` will fail with a clear `FileNotFoundError` listing which name patterns were checked.

If the data is in `.csv`, note that this format doesn't preserve data types as strictly as `.parquet`/`.pkl` — check `df.dtypes` after loading if needed.

## Training the model
```bash
python train.py
```

The script:

1. Loads `ga_sessions.parquet` and `ga_hits-001.parquet`, builds a binary target (`target`) from the presence of target events in a session.
2. Splits the data into train/test (70/30, stratified by target).
3. Trains the pipeline: feature engineering → data cleaning → grouping of rare categories (top-N) → clipping outliers in screen height → feature selection → OneHot/imputation → `DecisionTreeClassifier`.
4. Prints ROC-AUC on train and test.
5. Saves the model together with metadata to `ga_model.pkl` (serialized via `dill`, to preserve custom transformers).

Example output:

```
Loading GA Hits...
  Found file: ga_hits-001.parquet
GA Hits loaded.
Target built.
Loading GA Sessions...
  Found file: ga_sessions.parquet
GA Sessions: (XXXXXX, XX)
Final dataset: (XXXXXX, XX)
Target=1 share: 0.XXXX

Train: (XXXXXX, XX)
Test:  (XXXXXX, XX)

Training Decision Tree...
Training time: XX.X sec

Train ROC-AUC: 0.XXXX
Test ROC-AUC:  0.XXXX
ROC-AUC gap:   0.XXXX
Model saved: /path/to/ga_model.pkl
```
## Features used by the model
```text
Numeric: visit_number, has_keyword, is_russia, is_presence_city, visit_weekday, visit_hour, screen_width, screen_height.

Categorical (after grouping rare values into "other"): utm_medium, device_category, device_os, device_browser, geo_city_grouped, utm_source_grouped, utm_campaign_grouped, device_brand_grouped, utm_adcontent_grouped.

The fields session_id, client_id, visit_date, visit_time, device_screen_resolution, and the original (ungrouped) categorical columns are only used at intermediate stages and aren't fed into the model directly.
```
## Running the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs will be available at: `http://localhost:8000/docs`

### Endpoints

| Method | Path      | Description                                    |
|--------|-----------|-------------------------------------------------|
| GET    | /status   | Service health check                             |
| GET    | /version  | Model metadata (name, version, training date)    |
| POST   | /predict  | Predict conversion probability                    |

### Example `POST /predict` request
```json
{
  "session_id": "1234567890.1234567890",
  "client_id": "987654321.1234567890",
  "visit_date": "2021-10-15",
  "visit_time": "14:32:10",
  "visit_number": 1,
  "utm_source": "google",
  "utm_medium": "cpc",
  "utm_campaign": "spring_promo",
  "utm_adcontent": "banner_1",
  "utm_keyword": "car insurance",
  "device_category": "mobile",
  "device_os": "Android",
  "device_brand": "Samsung",
  "device_browser": "Chrome",
  "device_model": "SM-G960F",
  "geo_country": "Russia",
  "geo_city": "Moscow",
  "device_screen_resolution": "412x915"
}
```

### Example response
```json
{
  "session_id": "1234567890.1234567890",
  "prediction": 1,
  "probability": 0.7421
}
```

`prediction` — binary class (0/1), derived from `probability` at a 0.5 threshold. `probability` — the model's estimated probability of the target action.

## Notes
```text
All form fields except session_id, client_id, visit_date, visit_time, visit_number are optional (default None) — missing values are handled automatically by the pipeline.
main.py and train.py use the same transformer classes from pipeline.py, so data handling is identical during training and inference.
Loading raw data (load_data() in pipeline.py) isn't tied to a specific filename or format — you can freely change the event file number (-001, -002, ...) or format (.parquet/.pkl/.csv) without touching the code.
Model - DecisionTreeClassifier with class_weight="balanced", max_depth=10, min_samples_leaf=20 (depth/leaf constraints reduce overfitting given the strong class imbalance).
```
-e 
---
🇷🇺 [Читать на русском](https://github.com/ArturM99/conversion-prediction-service/tree/RU)
